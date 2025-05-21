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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset

# ViruLink imports
from ViruLink.train.OrdTri import OrdTri
from ViruLink.train.losses import CUM_TRE_LOSS, _adjacent_probs
from ViruLink.sampler.triangle_sampler import sample_triangles
from ViruLink.test.test_triangle import run_epoch
from ViruLink.setup.databases import database_info
from ViruLink.setup.run_parameters import parameters
from ViruLink.setup.score_profile import score_config
from ViruLink.search_utils import (
    DiamondCreateDB,
    DiamondSearchDB,
    CreateANISketchFolder,
    ANIDist)
from ViruLink.utils import (
    logging_header,
    read_fasta,
    get_file_path, 
    edge_list_to_presence_absence,
    compute_hypergeom_weights,
    create_graph)
from ViruLink.train.n2v import n2v
from ViruLink.relations.relationship_edges import build_relationship_edges



class TriDS(Dataset):
    def __init__(
        self,
        tris: List[Tuple],
        emb: Dict[str, np.ndarray],
        ani_df: pd.DataFrame,
        hyp_df: pd.DataFrame
    ):
        self.t = tris
        self.e = emb

        def lut(df):
            d = {}
            for s, t, w in df[["source", "target", "weight"]].itertuples(False):
                d[(s, t)] = w
                d[(t, s)] = w
            return d
        self.ani = lut(ani_df)
        self.hyp = lut(hyp_df)

    def _w(self, a, b, table):
        return torch.tensor([table.get((a, b), 0.0)], dtype=torch.float32)

    def __len__(self): return len(self.t)

    def __getitem__(self, i):
        q, r1, r2, b1, b2, b3 = self.t[i]
        eq = torch.tensor(self.e[q],  dtype=torch.float32)
        ea = torch.tensor(self.e[r1], dtype=torch.float32)
        eh = torch.tensor(self.e[r2], dtype=torch.float32)

        edge = torch.cat([
            self._w(q,  r1, self.ani), self._w(q,  r1, self.hyp),
            self._w(q,  r2, self.ani), self._w(q,  r2, self.hyp),
            self._w(r1, r2, self.ani), self._w(r1, r2, self.hyp)
        ])                       # shape [6]

        return {
            "eq": eq, "ea": ea, "eh": eh, "edge": edge,
            "lqa": torch.tensor(b1, dtype=torch.long),
            "lqh": torch.tensor(b2, dtype=torch.long),
            "lrr": torch.tensor(b3, dtype=torch.long),
        }





# ────────────────────────── helpers ──────────────────────────
def remove_version(df: pd.DataFrame):
    for col in ("source", "target"):
        df[col] = df[col].str.split(".").str[0]
    return df

def delete_tmp(tmp_path: str):
    """Remove temporary directory (ignore if already gone)."""
    if tmp_path and os.path.isdir(tmp_path):
        logging.info("Deleting temporary files in %s", tmp_path)
        shutil.rmtree(tmp_path, ignore_errors=True)


def make_global_excepthook(tmp_path: str):
    """Return a sys.excepthook that cleans up temp files first."""
    def _hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        delete_tmp(tmp_path)
        # call the original hook so default traceback still prints
        sys.__excepthook__(exc_type, exc, tb)
    return _hook


def make_signal_handler(tmp_path: str):
    """Return a signal handler that cleans up then exits."""
    def _handler(sig, frame):
        logging.info("Got signal %s – cleaning up temp files", sig)
        delete_tmp(tmp_path)
        sys.exit(128 + sig)
    return _handler


def validate_query(query_path: str):
    records = read_fasta(query_path)
    ids = [rec.id for rec in records]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate IDs found in query file – all IDs must be unique."
        )
    return ids
        
def get_paths_dict(databases_loc, classes_df):
    path_to_unprocs = {}
    
    for class_data in classes_df["Class"].to_list():
        class_unproc = f"{databases_loc}/{class_data}" #unprocessed in same folder
        logging.info(f"Checking for {class_data} unprocessed data.")
        
        if os.path.exists(class_unproc):
            logging.info(f"Found {class_data} unprocessed data at {class_unproc}")
            path_to_unprocs[class_data] = class_unproc
        
        else:
            logging.info(f"Could not find {class_data} unprocessed data at {class_unproc}")
            
    VOGDB_unproc = f"{databases_loc}/VOGDB" #unprocessed in same folder
    if os.path.exists(VOGDB_unproc):
        logging.info(f"Found VOGDB unprocessed data at {VOGDB_unproc}")
        path_to_unprocs["VOGDB"] = VOGDB_unproc
    else:
        logging.info(f"Could not find VOGDB unprocessed data at {VOGDB_unproc}")
        logging.info("Please download the VOGDB data.")
        sys.exit(1)
    
    return path_to_unprocs

def remove_version(df: pd.DataFrame):
    for col in ("source", "target"):
        df[col] = df[col].str.split(".").str[0]
    return df

# ──────────────────────── graph building ────────────────────────
def fuse_emb(ani_emb: Dict[str, np.ndarray],
             hyp_emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    fused = {}
    EMBED_DIM = parameters["embedding_dim"]
    for n in set(ani_emb) | set(hyp_emb):
        v1 = ani_emb.get(n, np.zeros(EMBED_DIM))
        v2 = hyp_emb.get(n, np.zeros(EMBED_DIM))
        fused[n] = np.concatenate([v1, v2])
    return fused

def process_graph(df: pd.DataFrame):
    df = df[df["source"] != df["target"]].copy()
    u = np.minimum(df["source"], df["target"])
    v = np.maximum(df["source"], df["target"])
    df[["u", "v"]] = np.column_stack([u, v])
    und   = df.groupby(["u", "v"], as_index=False)["weight"].max()
    max_w = df["weight"].max()
    nodes = pd.unique(df[["source", "target"]].values.ravel())
    self_loops = pd.DataFrame({"u": nodes, "v": nodes, "weight": max_w})
    rev = und.rename(columns={"u": "v", "v": "u"})
    return (pd.concat([und, rev, self_loops], ignore_index=True)
              .rename(columns={"u": "source", "v": "target"}))


def generate_database(class_data, unproc_path):
    db_outpath = f"{unproc_path}/{class_data}"
    DiamondCreateDB(get_file_path(unproc_path,"faa"), db_outpath, force=False)
    return db_outpath

def merge_presence_absence(db1: pd.DataFrame,
                           db2: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([db1, db2], axis=0, sort=False)
    merged = merged.fillna(0.0)
    merged = merged.astype("int8")
    return merged



def m8_processor(m8_file, class_data, eval_threshold, bitscore_threshold):
    columns = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]

    m8 = pd.read_csv(m8_file, sep="\t", header=None, names=columns)

    # Filter based on thresholds
    filtered_m8 = m8[(m8["evalue"] <= eval_threshold) & (m8["bitscore"] >= bitscore_threshold)]

    # Deduplicate by keeping the best hit
    filtered_m8 = filtered_m8.sort_values(["query", "evalue", "bitscore"], ascending=[True, True, False])
    filtered_m8 = filtered_m8.drop_duplicates(subset=["query", "target"], keep="first")

    # Create edge list
    edge_list = filtered_m8[["query", "target"]]
    
    # Create a presence-absence matrix using pandas pivot_table
    presence_absence_matrix = (edge_list
                               .assign(presence=1)  # Add a column with value 1 to indicate presence
                               .pivot_table(index="query", columns="target", values="presence", fill_value=0)
                              )
    
    # Optional: Reset the column names for cleaner formatting
    presence_absence_matrix.columns.name = None
    presence_absence_matrix.index.name = None
    return presence_absence_matrix

def build_rel_bounds(meta_df: pd.DataFrame,
                     rel_scores: Dict[str, int]) -> pd.DataFrame:
    df = build_relationship_edges(meta_df, rel_scores)
    return (df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
              .astype({"lower": "uint8", "upper": "uint8"})
              [["source", "target", "lower", "upper"]])
    
    

def ClassifyHandler(arguments, classes_df):
    """
    Build graphs, train OrdTri with a held-out validation split (Phase A),
    show its metrics + confusion matrix, then fine-tune the best checkpoint
    on the union of train + val (Phase B) before classifying the query
    sequences.  Neighbour search always uses *all* reference nodes (queries
    excluded).
    """

    # ───────────────────────────── 0 · housekeeping ─────────────────────────────
    qids     = validate_query(arguments.query)
    tmp_path = arguments.temp_dir
    os.makedirs(tmp_path, exist_ok=True)

    if not arguments.keep_temp:
        atexit.register(delete_tmp, tmp_path)
        sys.excepthook = make_global_excepthook(tmp_path)
        handler = make_signal_handler(tmp_path)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    # ───────────────────────────── 1 · graph construction ───────────────────────
    logging_header("Adding query sequences to HYP graph")
    db_params  = database_info()
    paths      = get_paths_dict(arguments.database_loc, classes_df)
    VOGDB_path = paths["VOGDB"]
    class_data = arguments.database

    meta = pd.read_csv(f"{arguments.database_loc}/{class_data}/{class_data}.csv")

    # 1.a  HYPERGEOMETRIC graph --------------------------------------------------
    VOGDB_dmnd = generate_database("VOGDB", VOGDB_path)
    m8_file    = DiamondSearchDB(VOGDB_dmnd, arguments.query, tmp_path,
                                 threads=arguments.threads, force=True)

    db_edge_txt = f"{arguments.database_loc}/{class_data}/edge_list.txt"
    pa_query    = m8_processor(m8_file, class_data, arguments.eval, arguments.bitscore)
    pa_db       = edge_list_to_presence_absence(db_edge_txt)
    merged_pa   = merge_presence_absence(pa_query, pa_db)

    w_mat = compute_hypergeom_weights(merged_pa, arguments.threads)
    src, dst, wts = create_graph(w_mat, threshold=0.0)
    hyp = pd.DataFrame({"source": src, "target": dst, "weight": wts})

    # 1.b  ANI graph -------------------------------------------------------------
    logging_header("Adding query sequences to ANI graph")
    q_sk_dir  = f"{tmp_path}/ANI_sketch"
    db_sk_dir = f"{arguments.database_loc}/{class_data}/ANI_sketch"
    skp         = db_params[db_params["Class"] == class_data]
    sketch_mode = skp["skani_sketch_mode"].iat[0]
    dist_mode   = skp["skani_dist_mode"].iat[0]

    q_sk_txt  = CreateANISketchFolder(arguments.query, q_sk_dir,
                                      arguments.threads, sketch_mode)
    db_sk_txt = f"{db_sk_dir}/sketches.txt"

    db_ani_path   = f"{arguments.database_loc}/{class_data}/self_ANI.tsv"
    q_db_ani_path = f"{tmp_path}/query_db_ANI.tsv"

    q_db_edges   = ANIDist(q_sk_txt, db_sk_txt, q_db_ani_path,
                           arguments.threads, dist_mode, False,
                           arguments.ANI_FRAC_weights)
    q_self_edges = ANIDist(q_sk_txt, q_sk_txt, f"{tmp_path}/self_ANI.tsv",
                           arguments.threads, dist_mode, False,
                           arguments.ANI_FRAC_weights)

    logging_header("Completing ANI graph")
    ani = pd.concat([pd.read_csv(db_ani_path, sep="\t"), q_db_edges, q_self_edges])

    # 1.c  Node2Vec embeddings ---------------------------------------------------
    ani = process_graph(remove_version(ani))
    hyp = process_graph(remove_version(hyp))
    ani_emb = n2v(ani, arguments.threads, parameters)
    hyp_emb = n2v(hyp, arguments.threads, parameters)
    emb     = fuse_emb(ani_emb, hyp_emb)

    qids_nover = {x.split('.')[0] for x in qids}
    nodes_in_graphs = [n for n in emb if n not in qids_nover]

    meta = meta.loc[meta["Accession"].isin(nodes_in_graphs)]
    rel  = build_rel_bounds(meta, score_config[class_data])

    # ───────────────────────────── 2 · OrdTri training ──────────────────────────
    RNG_SEED, EPOCHS, BATCH, LR = 42, 10, 512, 1e-3
    NUM_PER_CLASS = 4000
    EDGE_ORDER, EDGE_PRED_CNT = ("r1r2", "qr2", "qr1"), 3
    act = "relu" if not arguments.swiglu else "swiglu"
    EMBED_DIM = parameters["embedding_dim"]; COMB_DIM = EMBED_DIM * 2

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED); random.seed(RNG_SEED)

    LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config[class_data])}
    K_CLASSES  = max(LEVEL2RANK.values()) + 1
    RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}
    NR_CODE    = LEVEL2RANK["NR"]

    all_nodes = nodes_in_graphs.copy(); random.shuffle(all_nodes)
    cut = int(0.9 * len(all_nodes))
    train_nodes, val_nodes = all_nodes[:cut], all_nodes[cut:]

    def _sample(nodes, ncls):
        return sample_triangles(
            nodes,
            rel["source"].tolist(),
            rel["target"].tolist(),
            rel["lower"].astype("uint8").tolist(),
            rel["upper"].astype("uint8").tolist(),
            ncls, K_CLASSES, arguments.threads, RNG_SEED
        )

    # —— Phase A : train on 90 %, validate on 10 % —————————
    tri_train, tri_val = _sample(train_nodes, NUM_PER_CLASS), _sample(val_nodes, NUM_PER_CLASS // 4)
    ds_train, ds_val   = TriDS(tri_train, emb, ani, hyp), TriDS(tri_val, emb, ani, hyp)
    ld_train = DataLoader(ds_train, BATCH, shuffle=True)
    ld_val   = DataLoader(ds_val,   BATCH, shuffle=False)

    model = OrdTri(COMB_DIM, K_CLASSES, act=act).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    BEST_PATH = f"{tmp_path}/best_val.pt"
    best_valL = float("inf")

    logging_header("Phase A – training with held-out validation")
    for ep in range(1, EPOCHS + 1):
        trL, trH, trA, _ = run_epoch(model, ld_train, K_CLASSES, NR_CODE, opt,
                                     cpu_flag=arguments.cpu)
        vaL, vaH, vaA, _ = run_epoch(model, ld_val,   K_CLASSES, NR_CODE,
                                     cpu_flag=arguments.cpu)
        logging.info(
            "Ep%02d  train L=%.3f hit=%.3f acc=%.3f   "
            "val L=%.3f hit=%.3f acc=%.3f",
            ep, trL, trH, trA, vaL, vaH, vaA
        )
        if vaL < best_valL:
            best_valL = vaL
            torch.save(model.state_dict(), BEST_PATH)

    # Confusion matrix on held-out validation
    _, _, _, cm = run_epoch(model, ld_val, K_CLASSES, NR_CODE,
                            collect_cm=True, cpu_flag=arguments.cpu)
    if cm is not None:
        levels = list(score_config[class_data])
        logging_header("VALIDATION CONFUSION MATRIX  (held-out)")
        logging.info(pd.DataFrame(cm, index=levels, columns=levels).to_string())

    # —— Phase B : fine-tune on full set ————————————————
    logging_header("Phase B – fine-tuning on train + val")
    model.load_state_dict(torch.load(BEST_PATH))
    opt = torch.optim.Adam(model.parameters(), lr=LR * 0.1)   # tiny LR

    train_nodes = val_nodes = all_nodes                       # union
    tri_full    = _sample(all_nodes, NUM_PER_CLASS)
    ld_full     = DataLoader(TriDS(tri_full, emb, ani, hyp),
                             BATCH, shuffle=True)

    for ep in range(1, 4):   # a few fine-tune epochs
        trL, trH, trA, _ = run_epoch(model, ld_full, K_CLASSES, NR_CODE, opt,
                                     cpu_flag=arguments.cpu)
        logging.info("Full-fit Ep%02d  L=%.3f hit=%.3f acc=%.3f", ep, trL, trH, trA)

    # ───────────────────────────── 3 · helper LUTs & adjacency ────────────────
    def _lut(df):
        d = {}
        for s, t, w in df[["source", "target", "weight"]].itertuples(False):
            d[(s, t)] = w; d[(t, s)] = w
        return d
    ani_w, hyp_w = _lut(ani), _lut(hyp)

    def build_adj(tbl, k=10):
        adj = {}
        for (u, v), w in tbl.items():
            if u != v:
                adj.setdefault(u, []).append((w, v))
                adj.setdefault(v, []).append((w, u))
        return {n: [v for w, v in sorted(lst, key=lambda x: (-x[0], x[1]))[:k]]
                for n, lst in adj.items()}
    adj_hyp, adj_ani = build_adj(hyp_w), build_adj(ani_w)

    def _top_neigh(adj, q, banned, k=2):
        return [n for n in adj.get(q, []) if n not in banned][:k]

    def _predict(q, r1, r2):
        model.eval()
        with torch.no_grad():
            eq, ea, eh = (torch.tensor(emb[n], dtype=torch.float32, device=device)
                          for n in (q, r1, r2))
            edge = torch.tensor([
                ani_w.get((q,r1),0.0), hyp_w.get((q,r1),0.0),
                ani_w.get((q,r2),0.0), hyp_w.get((q,r2),0.0),
                ani_w.get((r1,r2),0.0), hyp_w.get((r1,r2),0.0)
            ], dtype=torch.float32, device=device)
            U = torch.full((1, K_CLASSES), 1.0 / K_CLASSES, device=device)
            cur = {"qr1": U.clone(), "qr2": U.clone(), "r1r2": U.clone()}
            for _ in range(EDGE_PRED_CNT):
                for key in EDGE_ORDER:
                    xa, xb = (eq, ea) if key=="qr1" else (eq, eh) if key=="qr2" else (ea, eh)
                    lg = model(xa.unsqueeze(0), xb.unsqueeze(0),
                               edge.unsqueeze(0),
                               torch.cat([cur["qr1"], cur["qr2"], cur["r1r2"]], 1))
                    cur[key] = _adjacent_probs(lg)
            r1_rank = int(torch.argmax(cur["qr1"])); r2_rank = int(torch.argmax(cur["qr2"]))
            return r1_rank, float(cur["qr1"][0,r1_rank]), \
                   r2_rank, float(cur["qr2"][0,r2_rank])

    UNKNOWN = {"", "nan", "na", "unknown"}
    def _unk(v): return pd.isna(v) or str(v).strip().lower() in UNKNOWN

    def _reconcile(row_pick, row_other, rank_pick, rank_other):
        levels = list(LEVEL2RANK.keys()); r = rank_other
        while r >= 0:
            v1, v2 = row_pick.get(levels[r], ""), row_other.get(levels[r], "")
            if _unk(v1) or _unk(v2) or v1 == v2: break
            r -= 1
        return rank_pick if r == rank_other else r

    # ───────────────────────────── 4 · classify each query ────────────────────
    bad_nodes = {x.split('.')[0] for x in qids}
    results   = []

    for q_full in qids:
        q = q_full.split('.')[0]

        hyp_nbrs = _top_neigh(adj_hyp, q, bad_nodes, 5)
        ani_nbrs = _top_neigh(adj_ani, q, bad_nodes, 5)
        r_hyp = hyp_nbrs[0] if hyp_nbrs else None
        r_ani = next((n for n in ani_nbrs if n != r_hyp), None)

        if r_hyp is None and len(ani_nbrs) >= 2:
            r_hyp, r_ani = ani_nbrs[:2]
        elif r_ani is None and len(hyp_nbrs) >= 2:
            r_hyp, r_ani = hyp_nbrs[:2]

        if r_hyp is None or r_ani is None or r_hyp == r_ani:
            pool = [n for n in all_nodes if n not in {q,r_hyp,r_ani} and n not in bad_nodes]
            if pool:
                if r_hyp is None: r_hyp = random.choice(pool)
                else:             r_ani = random.choice(pool)

        if r_hyp is None or r_ani is None or r_hyp == r_ani:
            logging.warning("Query %s skipped – neighbours not found", q_full)
            continue

        rank1, prob1, rank2, prob2 = _predict(q, r_hyp, r_ani)
        init_rel_1, init_rel_2 = RANK2LEVEL[rank1], RANK2LEVEL[rank2]

        if rank1 >= rank2:
            pick_node, pick_rank, pick_prob = r_hyp, rank1, prob1
            other_node, other_rank          = r_ani, rank2
        else:
            pick_node, pick_rank, pick_prob = r_ani, rank2, prob2
            other_node, other_rank          = r_hyp, rank1

        row_pick  = meta.loc[meta["Accession"] == pick_node ].squeeze()
        row_other = meta.loc[meta["Accession"] == other_node].squeeze()

        final_rank = _reconcile(row_pick, row_other, pick_rank, other_rank)
        final_rel  = RANK2LEVEL[final_rank]
        out_prob   = pick_prob if final_rank == pick_rank else "LOGIC"

        tax_values = {
            lvl: (row_pick[lvl] if LEVEL2RANK[lvl] <= final_rank and lvl in row_pick else "")
            for lvl in score_config[class_data]
        }

        results.append({
            "query"                       : q_full,
            "closest_node_1"              : r_hyp,
            "closest_node_2"              : r_ani,
            "initial_relationship_node1"  : init_rel_1,
            "initial_pred_prob_node1"     : prob1,
            "initial_relationship_node2"  : init_rel_2,
            "initial_pred_prob_node2"     : prob2,
            "choice_of_node"              : pick_node,
            "logical_based_final_relationship" : final_rel,
            "pred_prob"                   : out_prob,
            **tax_values,
        })

    # ───────────────────────────── 5 · output CSV ─────────────────────────────
    df = pd.DataFrame(results).drop(columns=["NR"], errors="ignore")
    logging_header("Classification results")
    logging.info(df.to_string(index=False))
    df.to_csv(arguments.output, index=False)



