#!/usr/bin/env python3
import atexit
import os
import shutil
import signal
import sys
import traceback
import logging
import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset # Dataset is defined locally as TriDS_classify
import yaml
import json
import gc
import psutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, NamedTuple

# ViruLink imports
from ViruLink.sampler.triangle_sampler import sample_triangles
from ViruLink.setup.databases import database_info # For DB-specific params
from ViruLink.search_utils import (
    DiamondCreateDB, DiamondSearchDB, 
    CreateANISketchFolder, ANIDist,
    m8_file_processor
)
from ViruLink.utils import (
    logging_header, read_fasta, get_file_path,
    edge_list_to_presence_absence, compute_hypergeom_weights, create_graph
)
from ViruLink.train.n2v import n2v
from ViruLink.relations.relationship_edges import build_relationship_edges
from ViruLink.default_yaml import default_yaml_dct
from ViruLink.train.losses import _adjacent_probs

# Imports from the MINIMAL train_utils.py
from ViruLink.train.train_utils import (
    run_epoch, initiate_OrdTriTwoStageAttn, 
    generate_score_profile_from_yaml,
    process_graph_for_n2v
)

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F



def _fix_n2v_cfg_classify(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = cfg.copy()
    if "window" not in out and "window_size" in out:
        out["window"] = out.pop("window_size")
    return out


def remove_node_versions_classify(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df.copy() if df is not None else pd.DataFrame()
    df_copy = df.copy()
    for col in ("source", "target"):
        if col in df_copy.columns and df_copy[col].dtype == 'object':
             df_copy[col] = df_copy[col].astype(str).str.split(".").str[0]
    return df_copy



def fuse_embeddings_classify(
    ani_emb: dict[str, np.ndarray],
    hyp_emb: dict[str, np.ndarray],
    embedding_dim: int
) -> dict[str, np.ndarray]:
    fused = {}
    all_nodes = set(ani_emb.keys()) | set(hyp_emb.keys())
    for n in all_nodes:
        v1 = ani_emb.get(n, np.zeros(embedding_dim, dtype=np.float32))
        v2 = hyp_emb.get(n, np.zeros(embedding_dim, dtype=np.float32))
        fused[n] = np.concatenate([v1, v2])
    return fused

class TriDS_classify(Dataset):
    def __init__(self, tris: list[tuple], emb: dict[str, np.ndarray],
                 processed_ani_df_for_lut: pd.DataFrame, 
                 processed_hyp_df_for_lut: pd.DataFrame, 
                 comb_dim: int) -> None:
        self.t = tris; self.e = emb; self.comb_dim = comb_dim
        if not tris: logging.warning("classify: TriDS_classify initialized with zero triangles.")

        def lut(df_lut: pd.DataFrame) -> dict[tuple[str, str], float]:
            d: dict[tuple[str, str], float] = {}
            if df_lut is None or df_lut.empty : return d
            if not all(col in df_lut.columns for col in ["source", "target", "weight"]):
                logging.warning(f"classify: TriDS_classify LUT: Missing columns. Has: {df_lut.columns}.")
                return d
            for s, t, w in df_lut[["source", "target", "weight"]].itertuples(index=False, name=None):
                try:
                    weight_val = float(w)
                    d[(str(s), str(t))] = weight_val
                    d[(str(t), str(s))] = weight_val
                except (ValueError, TypeError) as e:
                    logging.debug(f"classify: TriDS_classify LUT: Could not convert weight '{w}' for ({s}-{t}). Error: {e}. Defaulting to 0.0.")
                    d[(str(s), str(t))] = 0.0 
                    d[(str(t), str(s))] = 0.0
            return d
        self.ani_lut = lut(processed_ani_df_for_lut)
        self.hyp_lut = lut(processed_hyp_df_for_lut)

    def _get_weight(self, node_a: str, node_b: str, table: dict[tuple[str, str], float]) -> torch.Tensor:
        weight = table.get((str(node_a), str(node_b)), 0.0)
        if pd.isna(weight): weight = 0.0
        return torch.tensor([weight], dtype=torch.float32)

    def __len__(self) -> int: return len(self.t)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i >= len(self.t): raise IndexError(f"Index {i} out of bounds for TriDS_classify length {len(self.t)}.")
        q_node, r1_node, r2_node, bounds_qr1, bounds_qr2, bounds_r1r2 = self.t[i]
        default_emb = np.zeros(self.comb_dim, dtype=np.float32)
        str_q_node, str_r1_node, str_r2_node = str(q_node), str(r1_node), str(r2_node)

        eq_emb = torch.tensor(self.e.get(str_q_node, default_emb), dtype=torch.float32)
        ea_emb = torch.tensor(self.e.get(str_r1_node, default_emb), dtype=torch.float32)
        eh_emb = torch.tensor(self.e.get(str_r2_node, default_emb), dtype=torch.float32)
        
        triangle_raw_edge_feats = torch.cat([
            self._get_weight(str_q_node, str_r1_node, self.ani_lut),
            self._get_weight(str_q_node, str_r1_node, self.hyp_lut),
            self._get_weight(str_q_node, str_r2_node, self.ani_lut),
            self._get_weight(str_q_node, str_r2_node, self.hyp_lut),
            self._get_weight(str_r1_node, str_r2_node, self.ani_lut),
            self._get_weight(str_r1_node, str_r2_node, self.hyp_lut),
        ])
        
        return {"eq": eq_emb, "ea": ea_emb, "eh": eh_emb, "edge": triangle_raw_edge_feats,
                "lqa": torch.tensor(bounds_qr1, dtype=torch.long),
                "lqh": torch.tensor(bounds_qr2, dtype=torch.long),
                "lrr": torch.tensor(bounds_r1r2, dtype=torch.long)}

def build_rel_bounds_classify(
    meta_df: pd.DataFrame,
    level_to_rank_map: dict[str, int]
) -> pd.DataFrame:
    if meta_df.empty or not level_to_rank_map:
        return pd.DataFrame(columns=["source", "target", "lower", "upper"])
    if 'Accession' not in meta_df.columns:
        logging.error("classify: 'Accession' column missing from metadata for rel bounds.")
        return pd.DataFrame(columns=["source", "target", "lower", "upper"])
    df = build_relationship_edges(meta_df, level_to_rank_map) 
    
    if df.empty: return pd.DataFrame(columns=["source", "target", "lower", "upper"])
    return (df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
              .astype({"lower": "uint8", "upper": "uint8"})
              [["source", "target", "lower", "upper"]])


# --- helpers specific to classify.py (Kept local) ---
def delete_tmp(tmp_path: str):
    if tmp_path and os.path.isdir(tmp_path):
        logging.info("classify: Deleting temporary files in %s", tmp_path)
        try: shutil.rmtree(tmp_path)
        except Exception as e: logging.error(f"classify: Error deleting temp directory {tmp_path}: {e}")

def make_global_excepthook(tmp_path: str):
    original_excepthook = sys.excepthook
    def _hook(exc_type, exc, tb):
        logging.error("classify: Unhandled exception:", exc_info=(exc_type, exc, tb))
        delete_tmp(tmp_path)
        original_excepthook(exc_type, exc, tb)
    return _hook

def make_signal_handler(tmp_path: str):
    def _handler(sig, frame):
        sig_name = signal.strsignal(sig) if hasattr(signal, 'strsignal') else f"Signal {sig}"
        logging.info("classify: %s received. Cleaning temp files at %s and exiting.", sig_name, tmp_path)
        delete_tmp(tmp_path)
        sys.exit(128 + sig)
    return _handler

def validate_query(query_path: str):
    records = read_fasta(query_path)
    ids = [rec.id for rec in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs found in query file – all IDs must be unique.")
    return ids

def get_paths_dict_classify(databases_loc_str: str, classes_df_info: pd.DataFrame) -> dict:
    '''
    Get paths to unprocessed databases for classification.
    Args:
        databases_loc_str (str): Path to the databases location.
        classes_df_info (pd.DataFrame): DataFrame containing class information.
    Returns:
        dict: A dictionary mapping class names to their unprocessed paths.
    '''
    path_to_unprocs = {}
    for class_data_name in classes_df_info["Class"].unique().tolist():
        class_unproc_path = os.path.join(databases_loc_str, class_data_name)
        if os.path.exists(class_unproc_path): 
            
            # pretty output
            GREEN  = "\033[92m"
            BLUE = "\033[94m"
            RESET  = "\033[0m"
            msg = f"{GREEN}✔ {class_data_name:<20}{RESET} → {BLUE}Downloaded in {class_unproc_path}{RESET}"
            logging.info(msg)
            
            path_to_unprocs[class_data_name] = class_unproc_path
        else: logging.warning(f"classify: Path for DB class '{class_data_name}' not found: {class_unproc_path}")
    VOGDB_unproc_path = os.path.join(databases_loc_str, "VOGDB")
    if os.path.exists(VOGDB_unproc_path): path_to_unprocs["VOGDB"] = VOGDB_unproc_path
    else: logging.error(f"classify: VOGDB directory not found at {VOGDB_unproc_path}. Critical."); sys.exit(1)
    return path_to_unprocs

def generate_database_classify(class_data_name: str, unproc_path_db: str) -> str:
    faa_path = get_file_path(unproc_path_db, "faa", multi=False)
    if not faa_path:
        logging.error(f"classify: No .faa file found in {unproc_path_db} for Diamond DB. Aborting.")
        sys.exit(1)
    db_outpath_base = os.path.join(unproc_path_db, class_data_name)
    DiamondCreateDB(faa_path, db_outpath_base, force=False) 
    return db_outpath_base

def process_graph_for_n2v_classify(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["source", "target", "weight"])
    
    df_copy = df.copy()
    if "source" in df_copy.columns: df_copy["source"] = df_copy["source"].astype(str)
    if "target" in df_copy.columns: df_copy["target"] = df_copy["target"].astype(str)

    if 'weight' in df_copy.columns:
        df_copy['weight'] = pd.to_numeric(df_copy['weight'], errors='coerce')

    df_no_self_loops = df_copy[df_copy["source"] != df_copy["target"]].copy()

    if df_no_self_loops.empty:
        nodes = []
        if not df_copy.empty and "source" in df_copy.columns and "target" in df_copy.columns:
            nodes_pd_arr = pd.unique(df_copy[["source", "target"]].values.ravel('K'))
            nodes = nodes_pd_arr.tolist()
        max_w = 0.0
        if 'weight' in df_copy.columns and df_copy['weight'].notna().any():
            try:
                max_w_val = df_copy["weight"].max()
                if pd.notna(max_w_val): max_w = float(max_w_val)
            except TypeError: max_w = 0.0
    else:
        temp_df_no_self_loops = df_no_self_loops.copy()
        u_series = temp_df_no_self_loops["source"].astype(str)
        v_series = temp_df_no_self_loops["target"].astype(str)
        temp_df_no_self_loops.loc[:, "u"] = np.minimum(u_series, v_series)
        temp_df_no_self_loops.loc[:, "v"] = np.maximum(u_series, v_series)
        
        und = temp_df_no_self_loops.groupby(["u", "v"], as_index=False)["weight"].max()
        
        max_w_und = 0.0
        if not und.empty and 'weight' in und.columns and und['weight'].notna().any():
            try:
                max_w_und_val = und["weight"].max()
                if pd.notna(max_w_und_val): max_w_und = float(max_w_und_val)
            except TypeError: max_w_und = 0.0

        max_w_no_loops = 0.0
        if 'weight' in df_no_self_loops.columns and df_no_self_loops['weight'].notna().any():
            try:
                max_w_no_loops_val = df_no_self_loops["weight"].max()
                if pd.notna(max_w_no_loops_val): max_w_no_loops = float(max_w_no_loops_val)
            except TypeError: max_w_no_loops = 0.0
        
        max_w = max(max_w_und, max_w_no_loops)
        if pd.isna(max_w): max_w = 0.0

        nodes_pd_arr = pd.unique(df_no_self_loops[["source", "target"]].values.ravel('K'))
        nodes = nodes_pd_arr.tolist()

    self_loops_list = []
    if nodes:
        for node_val in nodes:
             self_loops_list.append({"u": node_val, "v": node_val, "weight": max_w})
    self_loops = pd.DataFrame(self_loops_list, columns=["u", "v", "weight"])

    final_df_parts = []
    if 'und' in locals() and not und.empty:
        final_df_parts.append(und)
        rev = und.rename(columns={"u": "v", "v": "u"})
        final_df_parts.append(rev)
    if not self_loops.empty:
        final_df_parts.append(self_loops)

    if not final_df_parts:
        return pd.DataFrame(columns=["source", "target", "weight"])
        
    final_df = pd.concat(final_df_parts, ignore_index=True)
    if final_df.empty: return pd.DataFrame(columns=["source", "target", "weight"])
    return final_df.rename(columns={"u": "source", "v": "target"})

def merge_presence_absence_classify(db1: pd.DataFrame, db2: pd.DataFrame) -> pd.DataFrame:
    if db1.empty and db2.empty: return pd.DataFrame()
    if db1.empty: return db2.fillna(0.0).astype("int8")
    if db2.empty: return db1.fillna(0.0).astype("int8")
    all_cols = db1.columns.union(db2.columns)
    db1_aligned = db1.reindex(columns=all_cols, index=db1.index, fill_value=0)
    db2_aligned = db2.reindex(columns=all_cols, index=db2.index, fill_value=0)
    merged = pd.concat([db1_aligned, db2_aligned], axis=0)
    return merged.fillna(0.0).astype("int8")

def m8_processor_classify(m8_file:str, eval_threshold:float, bitscore_thresh:float) -> pd.DataFrame:
    cols = ["query","target","pident","alnlen","mismatch","numgapopen","qstart","qend","tstart","tend","evalue","bitscore"]
    try: m8 = pd.read_csv(m8_file, sep="\t", header=None, names=cols)
    except pd.errors.EmptyDataError: return pd.DataFrame()
    if m8.empty: return pd.DataFrame()
    m8['pident'] = pd.to_numeric(m8['pident'],errors='coerce'); m8.dropna(subset=['pident'],inplace=True)
    filt_m8 = m8[(m8["evalue"]<=eval_threshold)&(m8["bitscore"]>=bitscore_thresh)]
    if filt_m8.empty: return pd.DataFrame()
    filt_m8=filt_m8.sort_values(["query","evalue","bitscore"],ascending=[True,True,False])
    edge_lst=filt_m8.drop_duplicates(subset=["query","target"],keep="first")[["query","target"]]
    if edge_lst.empty: return pd.DataFrame()
    edge_lst["query"]=edge_lst["query"].astype(str); edge_lst["target"]=edge_lst["target"].astype(str)
    pa_mat=(edge_lst.assign(presence=1).pivot_table(index="query",columns="target",values="presence",fill_value=0))
    pa_mat.columns.name=None; pa_mat.index.name=None; return pa_mat

def _create_edge_weight_lut(df_for_lut: pd.DataFrame) -> dict[tuple[str,str], float]:
    d_lut = {}
    if df_for_lut is not None and not df_for_lut.empty and \
       all(col in df_for_lut.columns for col in ["source", "target", "weight"]):
        for s, t, w in df_for_lut[["source", "target", "weight"]].itertuples(index=False, name=None):
            weight_val = 0.0
            try:
                weight_val = float(w)
                if pd.isna(weight_val): weight_val = 0.0
            except (ValueError, TypeError): weight_val = 0.0
            d_lut[(str(s), str(t))] = weight_val
            d_lut[(str(t), str(s))] = weight_val
    return d_lut

def _build_adj_for_prediction(edge_weight_lut: dict[tuple[str,str], float], k_neighbors=50) -> dict[str, list[str]]:
    adj = {}
    for (u_node, v_node), weight_val in edge_weight_lut.items():
        if u_node != v_node and pd.notna(weight_val):
            adj.setdefault(u_node, []).append((weight_val, v_node))
    return {node: [neighbor_node for _, neighbor_node in sorted(neighbor_list, key=lambda x: -x[0])[:k_neighbors]]
            for node, neighbor_list in adj.items()}

# --- Taxonomy Reconciliation (MODIFIED) ---
UNKNOWN_SET_PRED={"","nan","na","unknown",None} # Moved to global for reconcile_tax_final
def is_unknown_pred(val): # Moved to global for reconcile_tax_final
    return pd.isna(val) or str(val).strip().lower() in UNKNOWN_SET_PRED

def reconcile_tax_final(
    tax_primary: pd.Series,    # Taxonomy of the node that determined the effective_rank
    tax_secondary: pd.Series,  # Taxonomy of the other node
    effective_rank: int,       # The rank of the relationship to the primary node
    level_to_rank_map: Dict[str, int], # e.g., LEVEL2RANK
    nr_code: int               # e.g., NR_CODE
) -> Dict[str, str]: # Returns the final taxonomy dictionary for the query
    final_tax = {}
    # Iterate through taxonomic levels, sorted by their rank value (e.g., species, genus, family...)
    for lvl_name, lvl_rank_val in sorted(level_to_rank_map.items(), key=lambda item: item[1]):
        if lvl_rank_val <= effective_rank:
            # Value from the primary (chosen) reference node
            primary_value = tax_primary.get(lvl_name)
            
            if not is_unknown_pred(primary_value):
                final_tax[lvl_name] = str(primary_value)
            else:
                # Primary is unknown for this level.
                # If this level IS the "No Relationship" (NR) level, check secondary for NR.
                if lvl_rank_val == nr_code:
                    secondary_nr_value = tax_secondary.get(lvl_name)
                    if not is_unknown_pred(secondary_nr_value):
                        final_tax[lvl_name] = str(secondary_nr_value)
                    else: # Both primary and secondary NR are unknown
                        final_tax[lvl_name] = "" # Query's NR is also unknown
                else:
                    # For levels more specific than NR (e.g., genus, family):
                    # If the primary reference is "unknown" at this level,
                    # the query's corresponding tax level is also "unknown".
                    # We DO NOT fall back to the secondary reference for these specific levels.
                    final_tax[lvl_name] = "" 
        else:
            # For levels more specific than the effective_rank, taxonomy is considered unknown.
            final_tax[lvl_name] = ""
    return final_tax

# --- Main Handler ---
def ClassifyHandler(arguments: Any):
    #-----------------------------------------------------------------------------------#
    # LOAD CONFIGS and SETUP                                                            #
    #-----------------------------------------------------------------------------------#

    # Get correct YAML
    if arguments.config and os.path.exists(arguments.config):
        with open(arguments.config, 'r') as f: yaml_dict = yaml.safe_load(f)
        logging.info(f"classify: Loaded configuration from {arguments.config}")
    else:
        logging.info(f"classify: Config file not found or not provided. Using default.")
        yaml_dict = default_yaml_dct
    
    # Bind to variables
    config_mode = 'normal'
    mode_settings = yaml_dict['settings'][config_mode]
    training_params_yaml = mode_settings['training_params']
    model_params_yaml = training_params_yaml['Model']
    n2v_params_yaml_raw = mode_settings['n2v']
    yaml_levels_config = training_params_yaml['Levels']
    graph_making_config = mode_settings['graph_making']
    ani_cfg = graph_making_config.get('ANI', {})
    ani_prog = ani_cfg.get('ani_program', 'skani')

    # node2vec parameters
    current_n2v_parameters_fixed = _fix_n2v_cfg_classify(n2v_params_yaml_raw)
    current_n2v_parameters = {
        "walk_length": int(current_n2v_parameters_fixed['walk_length']),
        "p": float(current_n2v_parameters_fixed['p']), 
        "q": float(current_n2v_parameters_fixed['q']),
        "walks_per_node": int(current_n2v_parameters_fixed['walks_per_node']),
        "window": int(current_n2v_parameters_fixed['window']),
        "epochs": int(current_n2v_parameters_fixed['epochs']),
        "embedding_dim": int(current_n2v_parameters_fixed['embedding_dim'])
    }
    n2v_emb_dim = current_n2v_parameters["embedding_dim"]
    
    # database information and generating a score profile
    db_info_full_df = database_info() 
    all_db_names_list = db_info_full_df["Class"].unique().tolist()
    generated_score_config = generate_score_profile_from_yaml(
        yaml_levels_config, all_db_names_list
    )
    current_db_score_profile = generated_score_config.get(arguments.database)


    qids = validate_query(arguments.query)
    tmp_path = Path(arguments.temp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    original_excepthook_classify = sys.excepthook
    sys.excepthook = make_global_excepthook(str(tmp_path))
    atexit.register(delete_tmp, str(tmp_path))
    sig_handler_classify = make_signal_handler(str(tmp_path))
    for sig_val in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig_val, sig_handler_classify)
        except Exception as e: logging.warning(f"classify: Cannot set signal handler for {sig_val}: {e}")

    #-----------------------------------------------------------------------------------#
    # Graph Making                                                                      #
    #-----------------------------------------------------------------------------------#
    
    logging_header("Building Combined Graphs for N2V")
    
    # DB paths
    target_db_unproc_dir = Path(arguments.databases_loc) / arguments.database
    meta_csv_path = target_db_unproc_dir / f"{arguments.database}.csv"

    
    paths_classify = get_paths_dict_classify(arguments.databases_loc, db_info_full_df)
    VOGDB_unproc_dir = paths_classify.get("VOGDB")
    logging_header(VOGDB_unproc_dir)
        
    VOGDB_dmnd_db_base = generate_database_classify("VOGDB", VOGDB_unproc_dir)
    logging_header(VOGDB_dmnd_db_base)
    
    db_type = graph_making_config.get('hypergeometric', {}).get('protein_db', 'vogdb')
    if db_type=="custom_prot_db":
        logging.info("classify: Using custom protein database for hypergeometric graph making.")
        VOGDB_dmnd_db_base = target_db_unproc_dir / "custom_prot_db.dmnd"
        if not VOGDB_dmnd_db_base.exists():
            logging.error(f"classify: Custom protein database {VOGDB_dmnd_db_base} not found. Defaulting to VOGDB")
            db_type = graph_making_config.get('hypergeometric', {}).get('protein_db', 'vogdb')
    
    eval_thresh = graph_making_config.get('hypergeometric', {}).get('e_value', arguments.eval)
    bitscore_thresh = graph_making_config.get('hypergeometric', {}).get('bitscore', arguments.bitscore)
    dimaond_percent_id = graph_making_config.get('hypergeometric', {}).get('percent_id', None)
    diamond_db_cov = graph_making_config.get('hypergeometric', {}).get('db_cov', None)
    
    # Query VS VOGDB
    m8_q_vs_vog = DiamondSearchDB(VOGDB_dmnd_db_base, arguments.query, str(tmp_path), int(arguments.threads), True, percent_id=dimaond_percent_id, db_cov=diamond_db_cov)
    m8_q_vs_vog_mod = m8_file_processor(m8_q_vs_vog, eval_thresh, bitscore_thresh)
    pa_query_raw = edge_list_to_presence_absence(edge_list=m8_q_vs_vog_mod)
    
    # Remove version numbers
    pa_query = pd.DataFrame()
    if not pa_query_raw.empty:
        pa_q_temp = pa_query_raw.copy()
        pa_q_temp.index = pa_q_temp.index.astype(str).str.split('.').str[0]
        pa_q_temp.columns = pa_q_temp.columns.astype(str).str.split('.').str[0]
        pa_query = pa_q_temp
    
    
    db_edge_txt_path = target_db_unproc_dir / "edge_list.txt"
    pa_db_raw = edge_list_to_presence_absence(str(db_edge_txt_path))
    pa_db = pd.DataFrame()
    if not pa_db_raw.empty:
        pa_db_temp = pa_db_raw.copy()
        pa_db_temp.index = pa_db_temp.index.astype(str).str.split('.').str[0]
        pa_db_temp.columns = pa_db_temp.columns.astype(str).str.split('.').str[0]
        pa_db = pa_db_temp

    # Combine presence-absence matrices and compute hypergeometric weights between each pair of nodes
    merged_pa_combined = merge_presence_absence_classify(pa_query, pa_db)
    score_mat_combined = compute_hypergeom_weights(merged_pa_combined, int(arguments.threads), hypergeom=True)
    
    # Convert to 
    src_s, dst_s, wts_s = create_graph(score_mat_combined, 0.0)
    combined_hyp_edges_df = pd.DataFrame(columns=["source", "target", "weight"])
    combined_hyp_edges_df = pd.DataFrame({"source": src_s, "target": dst_s, "weight": wts_s})
    
    
    hyp_graph_combined_processed = process_graph_for_n2v_classify(
        remove_node_versions_classify(combined_hyp_edges_df)
    )
    logging.info(f"classify: Combined HYP graph (for N2V) processed: {hyp_graph_combined_processed.shape}")

    # Combined ANI Graph
    db_ani_fn = "mmseqs_ANI.tsv" if ani_prog == "mmseqs" else "self_ANI.tsv"
    db_ani_path = target_db_unproc_dir / db_ani_fn
    assert db_ani_path.exists(), f"Core DB ANI file {db_ani_path} not found."
    db_ani_edges_df_raw = pd.read_csv(db_ani_path, sep="\t")
    
    # making sure mmseqs columns are correctly mapped
    if ani_prog == "mmseqs":
        colmap_db = {}; db_ani_cols = db_ani_edges_df_raw.columns
        for c in ("query", "source"):
            if c in db_ani_cols: colmap_db[c] = "source"; break
        for c in ("target", "subject"):
            if c in db_ani_cols: colmap_db[c] = "target"; break
        for c in ("ani", "ANI", "pident", "identity"):
            if c in db_ani_cols: colmap_db[c] = "weight"; break
        if colmap_db: db_ani_edges_df_raw = db_ani_edges_df_raw.rename(columns=colmap_db)
    q_db_edges_df, q_self_edges_df = pd.DataFrame(columns=["source","target","weight"]), pd.DataFrame(columns=["source","target","weight"])
    
    # producing ANI graph
    if ani_prog == 'skani':
        q_sk_dir = tmp_path / "ANI_query_classify_sketches_combined"
        db_specific_params_row = db_info_full_df[db_info_full_df["Class"] == arguments.database].iloc[0] \
            if not db_info_full_df[db_info_full_df["Class"] == arguments.database].empty else pd.Series()
        skani_sketch_mode_cfg = ani_cfg.get('skani_sketch_mode_default', "")
        skani_sketch_mode = str(db_specific_params_row.get("skani_sketch_mode", skani_sketch_mode_cfg) or skani_sketch_mode_cfg).strip() # Ensure string
        skani_dist_mode_cfg = ani_cfg.get('skani_dist_mode_default', "")
        skani_dist_mode = str(db_specific_params_row.get("skani_dist_mode", skani_dist_mode_cfg) or skani_dist_mode_cfg).strip() # Ensure string

        len_w=bool(graph_making_config.get('ANI',).get('consider_alignment_length', arguments.ANI_FRAC_weights))
        q_query_skt = CreateANISketchFolder(arguments.query, str(q_sk_dir), int(arguments.threads), skani_sketch_mode)
        db_skt_path = target_db_unproc_dir / "ANI_sketch" / "sketches.txt"; assert db_skt_path.exists()
        q_db_edges_df=ANIDist(str(db_skt_path), q_query_skt, str(tmp_path/"q_db_skani_ANI_combined.tsv"), int(arguments.threads), skani_dist_mode, True, len_w)
        q_self_edges_df=ANIDist(q_query_skt, q_query_skt, str(tmp_path/"q_self_skani_ANI_combined.tsv"), int(arguments.threads), skani_dist_mode, True, len_w)
    
    elif ani_prog == 'mmseqs':
        from ViruLink.ani.ani_calc import m8_to_ani # Local import if not at top
        from ViruLink.search_utils import CreateDB as MMCreate, SearchDBs as MMSearch
        
        q_fasta=arguments.query
        q_mm_db_base=os.path.splitext(os.path.basename(q_fasta))[0]
        q_mm_db=str(tmp_path/f"{q_mm_db_base}_mm_combined")
        
        # Create MMSeqs DB for the query
        MMCreate(q_fasta,q_mm_db,type=2,force=True)
        tgt_db_fasta_path_str = get_file_path(str(target_db_unproc_dir),"fasta",False)
        
        tgt_mm_db_base=arguments.database
        tgt_mm_db=str(tmp_path/f"{tgt_mm_db_base}_mm_combined")
        
        MMCreate(tgt_db_fasta_path_str,tgt_mm_db,type=2,force=True)
        len_w=bool(graph_making_config.get('ANI',).get('consider_alignment_length', arguments.ANI_FRAC_weights))
        qdb_m8=MMSearch(q_mm_db,tgt_mm_db,str(tmp_path/"qdb_mm_search_combined"),str(tmp_path/"qdb_mm_tmp_combined"),int(arguments.threads),3, force=True)
        qdb_ani_out=str(tmp_path/"qdb_mm_ANI_combined.tsv"); m8_to_ani(qdb_m8,qdb_ani_out,int(arguments.threads),len_w)
        q_db_edges_df=pd.read_csv(qdb_ani_out,sep="\t").rename(columns={'query':'source','target':'target','ani':'weight'})
        qq_m8=MMSearch(q_mm_db,q_mm_db,str(tmp_path/"qq_mm_search_combined"),str(tmp_path/"qq_mm_tmp_combined"),int(arguments.threads),3, force=True)
        qq_ani_out=str(tmp_path/"qq_mm_ANI_combined.tsv"); m8_to_ani(qq_m8,qq_ani_out,int(arguments.threads),len_w)
        q_self_edges_df=pd.read_csv(qq_ani_out,sep="\t").rename(columns={'query':'source','target':'target','ani':'weight'})

    ani_df_edges_combined_all = pd.concat([db_ani_edges_df_raw, q_db_edges_df, q_self_edges_df], ignore_index=True)


    ani_graph_combined_processed = process_graph_for_n2v_classify(
        remove_node_versions_classify(ani_df_edges_combined_all)
    )
    print(ani_graph_combined_processed)
    
    # Legacy ANI weight rescaling (from working test_triangle.py) on the fully combined graph
    if model_params_yaml.get("rescale_ani_weights") and \
       "weight" in ani_graph_combined_processed.columns and \
       ani_graph_combined_processed['weight'].notna().any():
        logging.info("classify: Rescaling COMBINED ANI graph weights (Legacy Method).")
        w_comb = ani_graph_combined_processed["weight"].to_numpy(dtype=float)
        if not np.all(np.isnan(w_comb)): # Check if not all NaNs
            valid_w_comb = w_comb[~np.isnan(w_comb)]
            if valid_w_comb.size > 0:
                min_w_comb, max_w_comb = np.nanmin(valid_w_comb), np.nanmax(valid_w_comb)
                rng_comb = max_w_comb - min_w_comb
                temp_comb_graph_to_scale = ani_graph_combined_processed.copy()
                non_nan_mask_comb = ~np.isnan(w_comb)
                
                # Ensure assignment only to non-NaN positions
                scaled_values = np.empty_like(w_comb)
                if rng_comb > 1e-9:
                    scaled_values[non_nan_mask_comb] = (w_comb[non_nan_mask_comb] - min_w_comb) / rng_comb
                else:
                    # Check if max_w_comb itself is NaN (e.g., if valid_w_comb was empty, though caught by size check)
                    # This logic follows the one in working test_triangle.py
                    scaled_val_content = 1.0 if pd.notna(max_w_comb) and max_w_comb > 0.5 else 0.0
                    scaled_values[non_nan_mask_comb] = scaled_val_content
                
                # Assign back NaNs where they were original
                scaled_values[np.isnan(w_comb)] = np.nan 
                temp_comb_graph_to_scale.loc[:, "weight"] = scaled_values
                ani_graph_combined_processed = temp_comb_graph_to_scale
            else: logging.warning("classify: Combined ANI weights column empty after NaN filter. Skipping rescaling.")
        else: logging.warning("classify: Combined ANI weights column is all NaN. Skipping rescaling.")

    # --- 2. Single N2V run on combined graphs ---
    logging.info("classify: Running Node2Vec on FULLY COMBINED graphs.")
    combined_ani_emb = n2v(ani_graph_combined_processed, int(arguments.threads), current_n2v_parameters) \
        if not ani_graph_combined_processed.empty else {}
    combined_hyp_emb = n2v(hyp_graph_combined_processed, int(arguments.threads), current_n2v_parameters) \
        if not hyp_graph_combined_processed.empty else {}
    
    emb = fuse_embeddings_classify(combined_ani_emb, combined_hyp_emb, n2v_emb_dim)
    assert emb, "classify: No embeddings generated from combined graphs. Aborting."
    
    # --- 3. Setup for DB Training (using the global 'emb' but sampling only DB nodes) ---
    meta_db_main_raw = pd.read_csv(meta_csv_path)
    meta_db_main_raw["Accession"] = meta_db_main_raw["Accession"].astype(str)
    meta_db_for_training_ref = remove_node_versions_classify(meta_db_main_raw)
    
    qids_nover_set = {qid.split('.')[0] for qid in qids}
    db_nodes_for_training_pool = [
        acc for acc in meta_db_for_training_ref["Accession"].unique() 
        if acc in emb and acc not in qids_nover_set
    ]
    assert db_nodes_for_training_pool, "classify: No DB accessions from metadata found in combined embeddings for training."
    
    meta_train_val_nodes_ref = meta_db_for_training_ref[
        meta_db_for_training_ref["Accession"].isin(db_nodes_for_training_pool)
    ].copy()
    assert not meta_train_val_nodes_ref.empty, "classify: Metadata for DB training phases is empty."

    LEVEL2RANK = current_db_score_profile; K_CLASSES = max(LEVEL2RANK.values()) + 1
    RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}; NR_CODE = LEVEL2RANK["NR"]
    
    rel_bounds_for_db_training = build_rel_bounds_classify(meta_train_val_nodes_ref, LEVEL2RANK)

    RNG_SEED = training_params_yaml['RNG_seed']; EPOCHS = training_params_yaml['EPOCHS']
    BATCH_SIZE = training_params_yaml['BATCH']; LR = training_params_yaml['LEARNING_RATE']
    NUM_PER_CLASS_TRAIN = training_params_yaml.get('TRIANGLES_PER_CLASS_train', 4000)
    NUM_PER_CLASS_EVAL = training_params_yaml.get('TRIANGLES_PER_CLASS_eval', 1000)
    COMB_DIM_MODEL = n2v_emb_dim * 2

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED); random.seed(RNG_SEED)
    if device.type == 'cuda': torch.cuda.manual_seed_all(RNG_SEED)

    temp_db_nodes_A_shuffled = list(db_nodes_for_training_pool) # This is the list of DB nodes with embeddings
    random.Random(RNG_SEED).shuffle(temp_db_nodes_A_shuffled)
    
    split_idx_A = int(0.9 * len(temp_db_nodes_A_shuffled))
    if len(temp_db_nodes_A_shuffled) < 20 : split_idx_A = len(temp_db_nodes_A_shuffled)
    elif split_idx_A == len(temp_db_nodes_A_shuffled) and len(temp_db_nodes_A_shuffled) > 0 : split_idx_A -=1 
    
    train_nodes_A = temp_db_nodes_A_shuffled[:split_idx_A]
    val_nodes_A = temp_db_nodes_A_shuffled[split_idx_A:]
    assert train_nodes_A, "classify: No DB nodes for training set (Phase A)."
    num_dl_workers = max(0, int(arguments.threads) // 2)

    def _sample_tris_for_db_training_classify( nodes_list, rel_bounds_df, num_per_cls, seed_offset):
        if not nodes_list: logging.warning(f"classify _sample_tris_db: Node list empty."); return []
        if rel_bounds_df.empty: logging.warning(f"classify _sample_tris_db: Rel bounds empty."); return []
        src = rel_bounds_df["source"].astype(str).tolist(); tgt = rel_bounds_df["target"].astype(str).tolist()
        low = rel_bounds_df["lower"].astype(np.uint8).tolist(); upp = rel_bounds_df["upper"].astype(np.uint8).tolist()
        try:
            return sample_triangles(
                [str(n) for n in nodes_list], src, tgt, low, upp,
                int(num_per_cls), int(K_CLASSES), int(arguments.threads), int(RNG_SEED + seed_offset)
            )
        except Exception as e: logging.error(f"classify _sample_tris_db error: {e}"); return []

    train_rel_A = rel_bounds_for_db_training[
        rel_bounds_for_db_training["source"].isin(train_nodes_A) &
        rel_bounds_for_db_training["target"].isin(train_nodes_A)
    ].reset_index(drop=True)
    tri_train_A = _sample_tris_for_db_training_classify(train_nodes_A, train_rel_A, NUM_PER_CLASS_TRAIN, 0)
    assert tri_train_A, "classify: No training triangles for Phase A."
    # TriDS uses global 'emb', and LUTs from *combined* graphs, as model trains on DB nodes within this global context
    ds_train_A = TriDS_classify(tri_train_A, emb, ani_graph_combined_processed, hyp_graph_combined_processed, COMB_DIM_MODEL)
    ld_train_A = DataLoader(ds_train_A, BATCH_SIZE, shuffle=True, num_workers=num_dl_workers, pin_memory=(device.type=="cuda"))

    ld_val_A = None
    if val_nodes_A:
        val_rel_A = rel_bounds_for_db_training[
            rel_bounds_for_db_training["source"].isin(val_nodes_A) &
            rel_bounds_for_db_training["target"].isin(val_nodes_A)
        ].reset_index(drop=True)
        tri_val_A = _sample_tris_for_db_training_classify(val_nodes_A, val_rel_A, NUM_PER_CLASS_EVAL, 1)
        if tri_val_A:
            ds_val_A = TriDS_classify(tri_val_A, emb, ani_graph_combined_processed, hyp_graph_combined_processed, COMB_DIM_MODEL)
            ld_val_A = DataLoader(ds_val_A, BATCH_SIZE, shuffle=False, num_workers=num_dl_workers, pin_memory=(device.type=="cuda"))

    # --- Model Training Phase A (Train on DB train, Val on DB val) ---
    model = initiate_OrdTriTwoStageAttn(model_params=model_params_yaml, k_classes=K_CLASSES, n2v_embedding_dim=COMB_DIM_MODEL, device=device)
    optimizer_A = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=model_params_yaml.get('weight_decay', 1e-5))
    best_val_loss_A, BEST_MODEL_A_PATH = float("inf"), tmp_path / "best_model_A.pt"
    patience_A, early_stop_patience = 0, model_params_yaml.get('early_stopping_patience', 5)

    logging_header("classify: Phase A Training (DB train/val)")
    for ep in range(1, EPOCHS + 1):
        tl, th, ta, _ = run_epoch(model, ld_train_A, K_CLASSES, optimizer_A, device, model_params_yaml)
        vl, vh, va = float("nan"), float("nan"), float("nan"); current_metric_stop = float("inf")
        if ld_val_A:
            vl, vh, va, _ = run_epoch(model, ld_val_A, K_CLASSES, None, device, model_params_yaml, collect_cm=False)
            if not np.isnan(vl): current_metric_stop = vl
        else: 
            if not np.isnan(tl): current_metric_stop = tl
        logging.info(f"classify Phase A Ep {ep:02d}: T-L={tl:.4f} H={th:.3f} A={ta:.3f} | V-L={vl:.4f} H={vh:.3f} A={va:.3f}")
        if current_metric_stop < best_val_loss_A:
            best_val_loss_A = current_metric_stop; patience_A = 0; torch.save(model.state_dict(), str(BEST_MODEL_A_PATH))
            logging.info(f"classify Phase A: New best model saved (metric: {best_val_loss_A:.4f})")
        elif ld_val_A and not np.isnan(vl): patience_A += 1 # Increment patience only if val ran and loss is valid
        if ld_val_A and early_stop_patience > 0 and patience_A >= early_stop_patience:
            logging.info(f"classify Phase A: Early stopping at epoch {ep}."); break
    if not BEST_MODEL_A_PATH.exists() and EPOCHS > 0: torch.save(model.state_dict(), str(BEST_MODEL_A_PATH))
    assert BEST_MODEL_A_PATH.exists(), "classify: No model checkpoint after Phase A."
    model.load_state_dict(torch.load(str(BEST_MODEL_A_PATH), map_location=device))

    # --- Model Training Phase B (Finetune on ALL DB nodes) ---
    logging_header("classify: Phase B Training (Finetune on All DB Data)")
    all_db_nodes_for_B_finetune = temp_db_nodes_A_shuffled # Full list of DB nodes with embeddings
    num_finetune_epochs_B = model_params_yaml.get("finetune_epochs_after_val", 1)
    FINAL_MODEL_PATH = tmp_path / "final_classifier.pt"

    if num_finetune_epochs_B > 0 and all_db_nodes_for_B_finetune:
        # rel_bounds_for_db_training is already for all DB nodes in the pool
        tri_train_B = _sample_tris_for_db_training_classify(all_db_nodes_for_B_finetune, rel_bounds_for_db_training, NUM_PER_CLASS_TRAIN, 2) # seed offset 2
        if tri_train_B:
            ds_train_B = TriDS_classify(tri_train_B, emb, ani_graph_combined_processed, hyp_graph_combined_processed, COMB_DIM_MODEL)
            ld_train_B = DataLoader(ds_train_B, BATCH_SIZE, shuffle=True, num_workers=num_dl_workers, pin_memory=(device.type=="cuda"))
            optimizer_B = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=model_params_yaml.get('weight_decay',1e-5))
            for ep_b in range(1, num_finetune_epochs_B + 1):
                ft_l, ft_h, ft_a, _ = run_epoch(model, ld_train_B, K_CLASSES, optimizer_B, device, model_params_yaml)
                logging.info(f"classify Phase B Ep{ep_b:02d} Finetune L={ft_l:.4f} H={ft_h:.3f} A={ft_a:.3f}")
            torch.save(model.state_dict(), str(FINAL_MODEL_PATH))
        else: 
            logging.warning("classify: No triangles for Phase B. Using model from Phase A.")
            shutil.copy(str(BEST_MODEL_A_PATH), str(FINAL_MODEL_PATH))
    else:
        logging.info("classify: Skipping Phase B. Using model from Phase A.")
        shutil.copy(str(BEST_MODEL_A_PATH), str(FINAL_MODEL_PATH))
    
    model.load_state_dict(torch.load(str(FINAL_MODEL_PATH), map_location=device)); model.eval()
    
    # Prediction LUTs and Adjacency from COMBINED graphs
    ani_w_pred = _create_edge_weight_lut(ani_graph_combined_processed)
    hyp_w_pred = _create_edge_weight_lut(hyp_graph_combined_processed)
    adj_hyp_pred = _build_adj_for_prediction(hyp_w_pred); adj_ani_pred = _build_adj_for_prediction(ani_w_pred)
    best_hyp_n = {n:lst[0] for n,lst in adj_hyp_pred.items() if lst}; best_ani_n = {n:lst[0] for n,lst in adj_ani_pred.items() if lst}
    
    # Cosine similarity DB matrix from DB node embeddings *within the combined embedding space*
    db_nodes_for_cosine_matrix = [n for n in all_db_nodes_for_B_finetune if n in emb]
    hyp_mat_db_cos, ani_mat_db_cos=None,None; hyp_ids_for_cos_matrix, ani_ids_for_cos_matrix=[],[]
    if db_nodes_for_cosine_matrix: # Unchanged logic for cosine matrix
        hyp_parts_list,temp_hyp_ids=[],[]; ani_parts_list,temp_ani_ids=[],[]
        for n_cos in db_nodes_for_cosine_matrix:
            if emb[n_cos].shape[0]==COMB_DIM_MODEL:
                hyp_parts_list.append(emb[n_cos][n2v_emb_dim:]); temp_hyp_ids.append(n_cos)
                ani_parts_list.append(emb[n_cos][:n2v_emb_dim]); temp_ani_ids.append(n_cos)
        if hyp_parts_list:
            hyp_mat_stacked=np.stack(hyp_parts_list)
            if hyp_mat_stacked.size>0:
                norms=np.linalg.norm(hyp_mat_stacked,axis=1,keepdims=True); zero_norm=(norms<1e-12)
                norms[zero_norm]=1.0; hyp_mat_db_cos=hyp_mat_stacked/norms; hyp_mat_db_cos[zero_norm.flatten(),:]=0
                hyp_ids_for_cos_matrix=temp_hyp_ids
        if ani_parts_list:
            ani_mat_stacked=np.stack(ani_parts_list)
            if ani_mat_stacked.size>0:
                norms=np.linalg.norm(ani_mat_stacked,axis=1,keepdims=True); zero_norm=(norms<1e-12)
                norms[zero_norm]=1.0; ani_mat_db_cos=ani_mat_stacked/norms; ani_mat_db_cos[zero_norm.flatten(),:]=0
                ani_ids_for_cos_matrix=temp_ani_ids

    # --- Final Prediction ---
    _predict_final_memo = {}
    @torch.no_grad()
    def _predict_final(q_id,r1_id,r2_id): # Uses global 'emb' (fully combined)
        memo_key = tuple(sorted((q_id,r1_id,r2_id))); 
        if memo_key in _predict_final_memo: return _predict_final_memo[memo_key]
        model.eval()
        if not all(n in emb for n in [q_id,r1_id,r2_id]): _predict_final_memo[memo_key]=(NR_CODE,0.,NR_CODE,0.); return _predict_final_memo[memo_key]
        eq_t=torch.tensor(emb[q_id],dtype=torch.float32,device=device).unsqueeze(0); e1_t=torch.tensor(emb[r1_id],dtype=torch.float32,device=device).unsqueeze(0); e2_t=torch.tensor(emb[r2_id],dtype=torch.float32,device=device).unsqueeze(0)
        raw_f=torch.tensor([ani_w_pred.get((q_id,r1_id),0.),hyp_w_pred.get((q_id,r1_id),0.),ani_w_pred.get((q_id,r2_id),0.),hyp_w_pred.get((q_id,r2_id),0.),ani_w_pred.get((r1_id,r2_id),0.),hyp_w_pred.get((r1_id,r2_id),0.)],dtype=torch.float32,device=device).unsqueeze(0)
        unif_p=torch.full((1,K_CLASSES),1./K_CLASSES,device=device); prev_p=torch.cat([unif_p]*3,dim=1)
        la_l,lh_l=None,None; n_passes=model_params_yaml.get('max_recycles',0)+1; g_alpha=model_params_yaml.get('gate_alpha',0.1)
        for _p_idx in range(n_passes):
            _la,_lh,_lr,_,_,_=model(eq_t,e1_t,e2_t,raw_f,prev_p); la_l,lh_l=_la,_lh
            if _p_idx<n_passes-1:next_adj_p=torch.cat([_adjacent_probs(l.detach())for l in[_la,_lh,_lr]],dim=1); prev_p=(g_alpha*next_adj_p+(1.-g_alpha)*prev_p)
        if la_l is None or lh_l is None: _predict_final_memo[memo_key]=(NR_CODE,0.,NR_CODE,0.); return _predict_final_memo[memo_key]
        p_qr1=_adjacent_probs(la_l);r_qr1=torch.argmax(p_qr1,1).item();pr_qr1=p_qr1[0,r_qr1].item()
        p_qr2=_adjacent_probs(lh_l);r_qr2=torch.argmax(p_qr2,1).item();pr_qr2=p_qr2[0,r_qr2].item()
        _predict_final_memo[memo_key]=(r_qr1,pr_qr1,r_qr2,pr_qr2); return r_qr1,pr_qr1,r_qr2,pr_qr2
    
    logging_header("classify: Query Classification (Final Stage)")
    results_final_list = []
    valid_db_neigh_set = set(db_nodes_for_training_pool) # References must be from the original DB training pool

    for q_full_acc in qids:
        q_nov = q_full_acc.split('.')[0]; sel_h_src,sel_a_src="unknown","unknown"
        if q_nov not in emb: 
            logging.warning(f"classify: Query {q_full_acc} ({q_nov}) not in combined embeddings. Skipping."); 
            results_final_list.append({"query":q_full_acc,"error":"No query embedding in combined set"})
            continue
        r_h,r_a=None,None
        # Neighbor finding logic unchanged
        if q_nov in adj_hyp_pred:
            for cand in adj_hyp_pred[q_nov]:
                if cand in valid_db_neigh_set: r_h=cand; sel_h_src="hyp_adj"; break
        if q_nov in adj_ani_pred:
            for cand in adj_ani_pred[q_nov]:
                if cand in valid_db_neigh_set and cand!=r_h: r_a=cand; sel_a_src="ani_adj"; break
        if r_h is None and q_nov in best_hyp_n and best_hyp_n[q_nov] in valid_db_neigh_set: r_h=best_hyp_n[q_nov]; sel_h_src="best_edge_h"
        if r_a is None and q_nov in best_ani_n and best_ani_n[q_nov] in valid_db_neigh_set:
            if best_ani_n[q_nov]!=r_h:r_a=best_ani_n[q_nov];sel_a_src="best_edge_a"
            elif r_h is None:r_a=best_ani_n[q_nov];sel_a_src="best_edge_a_no_rh"
        if r_h is None and q_nov in emb and hyp_mat_db_cos is not None and len(hyp_ids_for_cos_matrix)>0:
            q_emb_f=emb[q_nov]
            if q_emb_f.shape[0]==COMB_DIM_MODEL:
                q_emb_h_p=q_emb_f[n2v_emb_dim:];q_v_h_n=np.linalg.norm(q_emb_h_p)
                if q_v_h_n>1e-9:
                    q_v_h_norm=q_emb_h_p/q_v_h_n;sims_h=hyp_mat_db_cos @ q_v_h_norm
                    for idx_s in np.argsort(-sims_h):
                        if idx_s<len(hyp_ids_for_cos_matrix): cand=hyp_ids_for_cos_matrix[idx_s]
                        if cand in valid_db_neigh_set:r_h=cand;sel_h_src="cosine_h";break
        if r_a is None and q_nov in emb and ani_mat_db_cos is not None and len(ani_ids_for_cos_matrix)>0:
            q_emb_f=emb[q_nov]
            if q_emb_f.shape[0]==COMB_DIM_MODEL:
                q_emb_a_p=q_emb_f[:n2v_emb_dim];q_v_a_n=np.linalg.norm(q_emb_a_p)
                if q_v_a_n>1e-9:
                    q_v_a_norm=q_emb_a_p/q_v_a_n;sims_a=ani_mat_db_cos @ q_v_a_norm
                    for idx_s in np.argsort(-sims_a):
                        if idx_s<len(ani_ids_for_cos_matrix): cand=ani_ids_for_cos_matrix[idx_s]
                        if cand in valid_db_neigh_set and cand!=r_h:r_a=cand;sel_a_src="cosine_a";break
        if r_h and not r_a: r_a=r_h; sel_a_src=f"reuse_h_as_a({sel_h_src})"
        elif r_a and not r_h: r_h=r_a; sel_h_src=f"reuse_a_as_h({sel_a_src})"
        if not r_h or not r_a: 
            logging.warning(f"classify: Query {q_full_acc} insufficient valid neighbors for final prediction. r_h:{r_h}, r_a:{r_a}."); 
            results_final_list.append({"query":q_full_acc,"error":"Insufficient valid neighbors for final prediction","rh_f":str(r_h),"ra_f":str(r_a),"sel_h":sel_h_src,"sel_a":sel_a_src})
            continue
        
        r1p, p1p, r2p, p2p = _predict_final(q_nov, r_h, r_a)
        
        # Determine primary reference and effective rank for taxonomy reconciliation
        effective_rank: int
        tax_primary_node_series: pd.Series
        tax_secondary_node_series: pd.Series
        chosen_node_for_rank: str
        final_probability: float

        if r1p >= r2p: # r_h (ref1) is primary or equally ranked
            effective_rank = r1p
            tax_primary_node_series = meta_train_val_nodes_ref[meta_train_val_nodes_ref["Accession"]==r_h].iloc[0] \
                if r_h in meta_train_val_nodes_ref["Accession"].values else pd.Series(dtype='object')
            tax_secondary_node_series = meta_train_val_nodes_ref[meta_train_val_nodes_ref["Accession"]==r_a].iloc[0] \
                if r_a in meta_train_val_nodes_ref["Accession"].values else pd.Series(dtype='object')
            chosen_node_for_rank = r_h
            final_probability = p1p
        else: # r_a (ref2) is primary
            effective_rank = r2p
            tax_primary_node_series = meta_train_val_nodes_ref[meta_train_val_nodes_ref["Accession"]==r_a].iloc[0] \
                if r_a in meta_train_val_nodes_ref["Accession"].values else pd.Series(dtype='object')
            tax_secondary_node_series = meta_train_val_nodes_ref[meta_train_val_nodes_ref["Accession"]==r_h].iloc[0] \
                if r_h in meta_train_val_nodes_ref["Accession"].values else pd.Series(dtype='object')
            chosen_node_for_rank = r_a
            final_probability = p2p
        
        final_tax_dict = reconcile_tax_final( # MODIFIED CALL with explicit primary/secondary tax
            tax_primary_node_series, tax_secondary_node_series, effective_rank, LEVEL2RANK, NR_CODE
        )
        final_rel_name = RANK2LEVEL.get(effective_rank, f"R_{effective_rank}")
        
        entry={
            "query":q_full_acc,
            "closest_node_1_r_hyp":r_h, "closest_node_2_r_ani":r_a,
            "neighbor_source_node1":sel_h_src, "neighbor_source_node2":sel_a_src,
            "ani_edge_to_node1":ani_w_pred.get((q_nov,r_h),np.nan),
            "hyp_edge_to_node1":hyp_w_pred.get((q_nov,r_h),np.nan),
            "ani_edge_to_node2":ani_w_pred.get((q_nov,r_a),np.nan),
            "hyp_edge_to_node2":hyp_w_pred.get((q_nov,r_a),np.nan),
            "initial_relationship_to_node1":RANK2LEVEL.get(r1p, f"R{r1p}"), # Individual prediction to ref1
            "initial_pred_prob_to_node1":f"{p1p:.4f}",
            "initial_relationship_to_node2":RANK2LEVEL.get(r2p, f"R{r2p}"), # Individual prediction to ref2
            "initial_pred_prob_to_node2":f"{p2p:.4f}",
            "chosen_node_for_final_rank":chosen_node_for_rank,
            "final_relationship":final_rel_name,
            "final_pred_prob":f"{final_probability:.4f}"
        }
        for lk in LEVEL2RANK:entry[lk]=final_tax_dict.get(lk,"")
        results_final_list.append(entry)

    # Output formatting unchanged
    results_df_out=pd.DataFrame(results_final_list)
    if "NR" in results_df_out.columns:results_df_out=results_df_out.drop(columns=["NR"],errors="ignore")
    fixed_out_cols=["query","closest_node_1_r_hyp","closest_node_2_r_ani","neighbor_source_node1","neighbor_source_node2","ani_edge_to_node1","hyp_edge_to_node1","ani_edge_to_node2","hyp_edge_to_node2","initial_relationship_to_node1","initial_pred_prob_to_node1","initial_relationship_to_node2","initial_pred_prob_to_node2","chosen_node_for_final_rank","final_relationship","final_pred_prob"]
    sorted_level_names=sorted(LEVEL2RANK.keys(),key=lambda k:LEVEL2RANK[k]);tax_out_cols=[l for l in sorted_level_names if l!="NR"]
    error_col_keys = ["error","rh_f","ra_f","sel_h","sel_a"]
    error_col = [key for key in error_col_keys if key in results_df_out.columns]
    temp_ordered_cols=fixed_out_cols+tax_out_cols+error_col;final_ordered_cols=[c for c in temp_ordered_cols if c in results_df_out.columns]
    for c in results_df_out.columns:
        if c not in final_ordered_cols:final_ordered_cols.append(c)
    if not results_df_out.empty:results_df_out=results_df_out[final_ordered_cols]
    else:results_df_out=pd.DataFrame(columns=final_ordered_cols)
    logging_header("classify: Classification Results (final output)")
    if not results_df_out.empty :
        cols_to_check_for_data = [col for col in results_df_out.columns if col not in ["query"] + error_col_keys]
        meaningful_data_exists = False
        if cols_to_check_for_data: meaningful_data_exists = results_df_out[cols_to_check_for_data].notna().any().any()
        if meaningful_data_exists: logging.info(f"\n{results_df_out.to_string(index=False)}")
        elif not results_df_out.empty : logging.info(f"classify: Results only query/errors.\n{results_df_out[['query'] + error_col].to_string(index=False)}")
        results_df_out.to_csv(arguments.output,index=False);logging.info(f"classify: Output saved to {arguments.output}")
    else: 
        pd.DataFrame(columns=fixed_out_cols[:1]).to_csv(arguments.output,index=False) 
        logging.info(f"classify: No results. Empty output file: {arguments.output}")
    
    if not arguments.keep_temp:delete_tmp(str(tmp_path))
    else:logging.info(f"classify: Temporary files kept: {tmp_path}")
    sys.excepthook=original_excepthook_classify

