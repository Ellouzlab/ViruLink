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
    
    


# ────────────────────────── main entry ──────────────────────────
def ClassifyHandler(arguments, classes_df):

    qids = validate_query(arguments.query)
    tmp_path = arguments.temp_dir
    os.makedirs(tmp_path, exist_ok=True)

    # ---------- install cleanup hooks ----------
    if not arguments.keep_temp:
        # run on normal exit
        atexit.register(delete_tmp, tmp_path)

        # run on unhandled exception
        sys.excepthook = make_global_excepthook(tmp_path)

        # run on Ctrl-C or kill <sigterm>
        handler = make_signal_handler(tmp_path)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    # ---------- Part 1 – add query to graphs ----------
    # Setup
    logging_header("Adding query sequences to HYP graph")
    database_parameter = database_info()
    paths_to_unprocs = get_paths_dict(arguments.database_loc, classes_df)
    VOGDB_path = paths_to_unprocs["VOGDB"]
    class_data = arguments.database
    meta = pd.read_csv(f"{arguments.database_loc}/{class_data}/{class_data}.csv")
    print(meta)
    
    # Blast against VOGDB
    logging.info("blasting query against VOGDB")
    VOGDB_dmnd = generate_database("VOGDB", VOGDB_path)
    m8_file = DiamondSearchDB(
        VOGDB_dmnd,
        arguments.query,
        tmp_path,
        threads=arguments.threads,
        force=True
    )
    # Process m8 file
    db_edge_list_path = f"{arguments.database_loc}/{class_data}/edge_list.txt"

    # generate presence/absence matrix
    pa_query = m8_processor(m8_file, class_data, arguments.eval, arguments.bitscore)
    pa_db = edge_list_to_presence_absence(db_edge_list_path)
    merged_pa = merge_presence_absence(pa_query, pa_db)
    no_proteins_genomes = pa_query.index[(pa_query == 0).all(axis=1)].tolist() # For the user
    
    
    # calculate hypergeometric edges
    w_mat = compute_hypergeom_weights(merged_pa, arguments.threads)
    sources, destinations, weights = create_graph(w_mat, threshold=0.0)
    hyp = pd.DataFrame({"source": sources, "target": destinations, "weight": weights})
    
    # ANI graph building
    logging_header("Adding query sequences to ANI graph")
    query_ANI_sketch_folder = f"{tmp_path}/ANI_sketch"
    db_ANI_sketch_folder = f"{arguments.database_loc}/{class_data}/ANI_sketch"
    
    # Get modes
    skani_parameters = database_parameter[database_parameter["Class"]==class_data]
    skani_sketch_mode = skani_parameters["skani_sketch_mode"].values[0]
    skani_dist_mode = skani_parameters["skani_dist_mode"].values[0]
    
    # Get/generate sketches
    query_sketch_paths_txt = CreateANISketchFolder(arguments.query, query_ANI_sketch_folder, arguments.threads, skani_sketch_mode)
    db_sketch_paths_txt = f"{db_ANI_sketch_folder}/sketches.txt"
    
    db_ANI_edges_path = f"{arguments.database_loc}/{class_data}/self_ANI.tsv"
    query_db_ANI_edges_path = f"{tmp_path}/query_db_ANI.tsv"
    query_db_ANI_edges = ANIDist(
        query_sketch_paths_txt,
        db_sketch_paths_txt,
        query_db_ANI_edges_path,
        arguments.threads,
        skani_dist_mode,
        False,
        arguments.ANI_FRAC_weights
    )
    query_self_ANI_edges = ANIDist(
        query_sketch_paths_txt,
        query_sketch_paths_txt,
        f"{tmp_path}/self_ANI.tsv",
        arguments.threads,
        skani_dist_mode,
        False,
        arguments.ANI_FRAC_weights
    )
    
    logging_header("Completing ANI Graph")
    # Read the edges
    db_ANI_edges = pd.read_csv(db_ANI_edges_path, sep="\t")
    ani = pd.concat(
        [
            db_ANI_edges,
            query_db_ANI_edges,
            query_self_ANI_edges
        ]
    )
    
    logging_header("Running Node2vec")
    ani = process_graph(remove_version(ani))
    hyp = process_graph(remove_version(hyp))
    ani_emb = n2v(ani,arguments.threads, parameters)
    hyp_emb = n2v(hyp,arguments.threads, parameters)
    emb = fuse_emb(ani_emb, hyp_emb)
    nodes_in_graphs = emb.keys()
    
    qids_conversion_dict = {x.split('.')[0]: x for x in qids}
    nodes_in_graphs = [x for x in nodes_in_graphs if x not in qids_conversion_dict.keys()]
    
    meta = meta.loc[meta["Accession"].isin(nodes_in_graphs)]
    rel = build_rel_bounds(meta, score_config[class_data])
    
    
    
    
    # ───────────────────── train OrdTri on 90 / 10 split ─────────────────────
    # hyper-parameters copied verbatim from test_triangle.py
    RNG_SEED      = 42
    EPOCHS        = 10
    BATCH_SIZE    = 512
    LR            = 1e-3
    NUM_PER_CLASS = 4000
    LAMBDA_INT    = 1.0
    LAMBDA_TRI    = 0
    EDGE_ORDER    = ("r1r2", "qr2", "qr1")   # same prediction schedule
    EDGE_PRED_CNT = 3
    act = "relu" if not arguments.swiglu else "swiglu"
    EMBED_DIM = parameters["embedding_dim"]
    COMB_DIM  = EMBED_DIM * 2                # ANI ⨁ HYP embeddings

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    logging.info("Using device: %s", device)
    logging.info("Using %d threads", arguments.threads)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    # rank helpers (flexible to any score_config edits)
    LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config[class_data])}
    K_CLASSES  = max(LEVEL2RANK.values()) + 1
    RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}
    NR_CODE    = LEVEL2RANK["NR"]

    # split DB nodes (queries are excluded)
    all_nodes = list(nodes_in_graphs)
    random.shuffle(all_nodes)
    cut = int(0.9 * len(all_nodes))
    train_nodes, val_nodes = all_nodes[:cut], all_nodes[cut:]

    edges_train = rel[rel["source"].isin(train_nodes) &
                    rel["target"].isin(train_nodes)].reset_index(drop=True)
    edges_val   = rel[rel["source"].isin(val_nodes) &
                    rel["target"].isin(val_nodes)].reset_index(drop=True)

    def _sample(nodes, rel_df, n_per_cls, threads):
        return sample_triangles(
            nodes,
            rel_df["source"].tolist(),
            rel_df["target"].tolist(),
            rel_df["lower"].astype("uint8").tolist(),
            rel_df["upper"].astype("uint8").tolist(),
            n_per_cls,
            K_CLASSES,
            threads,
            RNG_SEED
        )

    tri_train = _sample(train_nodes, edges_train, NUM_PER_CLASS, arguments.threads)
    tri_val   = _sample(val_nodes,   edges_val,   NUM_PER_CLASS // 4, arguments.threads)

    ds_train = TriDS(tri_train, emb, ani, hyp)
    ds_val   = TriDS(tri_val,   emb, ani, hyp)
    ld_train = DataLoader(ds_train, BATCH_SIZE, shuffle=True)
    ld_val   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False)

    model = OrdTri(COMB_DIM, K_CLASSES, act=act).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    logging_header("Training OrdTri edge predictor")
    for ep in range(1, EPOCHS + 1):
        trL, trH, trA, _ = run_epoch(
            model, ld_train,
            K_CLASSES,           # total number of rank levels
            NR_CODE,             # index of the “NR” level
            opt,
            cpu_flag=arguments.cpu)
        vaL, vaH, vaA, _ = run_epoch(model, ld_val, K_CLASSES, NR_CODE, cpu_flag=arguments.cpu)
        logging.info(
            "Ep%02d  train L=%.3f hit=%.3f acc=%.3f   val L=%.3f hit=%.3f acc=%.3f",
            ep, trL, trH, trA, vaL, vaH, vaA
        )

    # ─────────── confusion matrix for point-label edges (val) ───────────
    _, _, _, val_cm = run_epoch(model, ld_val, K_CLASSES, NR_CODE,
                            collect_cm=True, cpu_flag=arguments.cpu)
    if val_cm is not None:
        levels = list(score_config[class_data])
        logging_header("VALIDATION CONFUSION MATRIX  (point-label edges only)")
        print(pd.DataFrame(val_cm, index=levels, columns=levels).to_string())
    # ─────────────────────────────────────────────────────────────────────
    
    # ─────────── helpers for reconciliation ───────────
    UNKNOWN_TOKENS = {"", "nan", "na", "unknown"}

    def _is_unknown(val) -> bool:
        """True if *val* is missing or any flavour of ‘unknown’."""
        if pd.isna(val):
            return True
        return str(val).strip().lower() in UNKNOWN_TOKENS

    def _first_known(*vals):
        """Return the first non-unknown value (or '' if none)."""
        for v in vals:
            if not _is_unknown(v):
                return v
        return ""

    def _reconcile_taxa(row1: pd.Series,
                        row2: pd.Series,
                        starting_rank: int) -> int:
        """
        Walk *up* the taxonomy until the two neighbours agree (ignoring unknowns).

        Parameters
        ----------
        row1, row2 : taxonomy rows for the two reference viruses
        starting_rank : numeric rank picked by OrdTri (e.g. species = 4)

        Returns
        -------
        int
            The reconciled rank (may be == starting_rank if no conflict).
        """
        # ranks are ordered broad → specific in score_config
        levels = list(score_config[class_data])          # e.g. ["family","genus","species"]
        for r in range(starting_rank, -1, -1):           # walk species → genus → …
            lvl = levels[r]
            v1, v2 = row1.get(lvl, ""), row2.get(lvl, "")
            if _is_unknown(v1) or _is_unknown(v2) or v1 == v2:
                return r                                 # match (or unknown) ⇒ stop here
        return 0                                         # fall back to top level




    # edge-weight LUTs for fast lookup
    def _lut(df):
        d = {}
        for s, t, w in df[["source", "target", "weight"]].itertuples(False):
            d[(s, t)] = w
            d[(t, s)] = w
        return d
    ani_w = _lut(ani)
    hyp_w = _lut(hyp)

    def _top_neighbours(weight_tbl, q, banned, k=2):
        """
        Return up to *k* distinct neighbours of q with largest weight.
        Self-loops and any node in *banned* are ignored.
        """
        scores = {}
        for (u, v), w in weight_tbl.items():
            if u == q and v not in banned and v != q:
                scores[v] = max(scores.get(v, -1.0), w)
            elif v == q and u not in banned and u != q:
                scores[u] = max(scores.get(u, -1.0), w)
        # sort once → highest weight first
        return sorted(scores, key=scores.get, reverse=True)[:k]
    def _edge_vec(a, b):
        return torch.tensor([
            ani_w.get((a, b), 0.0), hyp_w.get((a, b), 0.0)
        ], dtype=torch.float32, device=device).repeat(3)  # expands to 6 elems

    def _predict(q, r1, r2):
        """
        Return (rank1, prob1, rank2, prob2) for edges q-r1 and q-r2.
        """
        model.eval()
        with torch.no_grad():
            eq = torch.tensor(emb[q],  dtype=torch.float32, device=device)
            ea = torch.tensor(emb[r1], dtype=torch.float32, device=device)
            eh = torch.tensor(emb[r2], dtype=torch.float32, device=device)

            edge = torch.tensor([
                ani_w.get((q,  r1), 0.0), hyp_w.get((q,  r1), 0.0),
                ani_w.get((q,  r2), 0.0), hyp_w.get((q,  r2), 0.0),
                ani_w.get((r1, r2), 0.0), hyp_w.get((r1, r2), 0.0)
            ], dtype=torch.float32, device=device)

            # uniform priors
            U = torch.full((1, K_CLASSES), 1.0 / K_CLASSES, device=device)
            cur_p = {"qr1": U.clone(), "qr2": U.clone(), "r1r2": U.clone()}

            for _ in range(EDGE_PRED_CNT):
                for key in EDGE_ORDER:
                    if key == "qr1":   xa, xb = eq.unsqueeze(0), ea.unsqueeze(0)
                    elif key == "qr2": xa, xb = eq.unsqueeze(0), eh.unsqueeze(0)
                    else:              xa, xb = ea.unsqueeze(0), eh.unsqueeze(0)

                    lg = model(
                        xa, xb, edge.unsqueeze(0),
                        torch.cat([cur_p["qr1"], cur_p["qr2"], cur_p["r1r2"]], dim=1)
                    )
                    cur_p[key] = _adjacent_probs(lg)

            r1_rank = int(torch.argmax(cur_p["qr1"]).item())
            r2_rank = int(torch.argmax(cur_p["qr2"]).item())
            r1_prob = float(cur_p["qr1"][0, r1_rank].item())
            r2_prob = float(cur_p["qr2"][0, r2_rank].item())
            return r1_rank, r1_prob, r2_rank, r2_prob

    results = []
    bad_nodes = set(qids_conversion_dict.keys())  # exclude ALL queries everywhere

    for q_full in qids:                  # keep original id for reporting
        q = q_full.split(".")[0]         # version-less ID used in graphs

        # candidate lists (grab a few extra in case of ties)
        hyp_nbrs = _top_neighbours(hyp_w, q, bad_nodes, k=5)
        ani_nbrs = _top_neighbours(ani_w, q, bad_nodes, k=5)

        # ── pick top-HYP first (if it exists) ──
        r_hyp = hyp_nbrs[0] if hyp_nbrs else None

        # ── pick the best ANI neighbour that differs from r_hyp ──
        r_ani = next((n for n in ani_nbrs if n != r_hyp), None)

        # ── fall-back #1 : one list empty → take top-2 from the other ──
        if r_hyp is None and len(ani_nbrs) >= 2:
            r_hyp, r_ani = ani_nbrs[:2]
        elif r_ani is None and len(hyp_nbrs) >= 2:
            r_hyp, r_ani = hyp_nbrs[:2]

        # ── fall-back #2 : still only one unique neighbour → random DB node ──
        if r_hyp is None or r_ani is None or r_hyp == r_ani:
            pool = [n for n in all_nodes if n not in {q, r_hyp, r_ani} and n not in bad_nodes]
            if pool:
                if r_hyp is None:
                    r_hyp = random.choice(pool)
                else:
                    r_ani = random.choice(pool)

        # ── final sanity-check ──
        if r_hyp is None or r_ani is None or r_hyp == r_ani:
            logging.warning("Query %s could not find two distinct neighbours – skipped", q_full)
            continue

        # the triangle for prediction
        r1, r2 = r_hyp, r_ani
        
              
        rank1, prob1, rank2, prob2 = _predict(q, r1, r2)

        # ------------------------------------------------------------------
        # new logic that reconciles the two candidate edges
        # ------------------------------------------------------------------
        row1 = meta.loc[meta["Accession"] == r1].squeeze()
        row2 = meta.loc[meta["Accession"] == r2].squeeze()

        # pick the edge OrdTri judged stronger (tie-break by prob)
        if rank1 >= rank2:
            picked_node, picked_rank, picked_prob = r1, rank1, prob1
        else:
            picked_node, picked_rank, picked_prob = r2, rank2, prob2

        # ------------------------------------------------------------------
        # logical reconciliation – no probability tie-breaks
        # ------------------------------------------------------------------
        row1, row2 = (
            meta.loc[meta["Accession"] == r1].squeeze(),
            meta.loc[meta["Accession"] == r2].squeeze(),
        )

        # default to r1 unless its taxon is unknown at the final rank
        initial_pick = r1
        initial_rank = rank1
        initial_prob = prob1

        # compute the reconciled (possibly broader) rank
        resolved_rank = _reconcile_taxa(row1, row2, min(rank1, rank2))

        # if initial pick is unknown at the resolved rank, switch to r2
        lvl_name = list(score_config[class_data])[resolved_rank]
        if _is_unknown(row1.get(lvl_name, "")):
            initial_pick, initial_rank, initial_prob = r2, rank2, prob2

        picked_node  = initial_pick
        picked_rank  = resolved_rank
        picked_prob  = initial_prob if resolved_rank == initial_rank else "LOGIC"
        # ------------------------------------------------------------------


        # taxonomy columns up to the predicted level
        tax_row = meta.loc[meta["Accession"] == picked_node].squeeze() \
                  if "Accession" in meta.columns else pd.Series(dtype=str)
        tax_values = {
            lvl: (tax_row[lvl] if LEVEL2RANK[lvl] <= picked_rank and lvl in tax_row else "")
            for lvl in score_config[class_data]
        }

        results.append({
            "query":        q_full,
            "closest_node": picked_node,
            "relationship": RANK2LEVEL[picked_rank],
            "pred_prob":    picked_prob,
            **tax_values,
        })



    classif_df = pd.DataFrame(results)
    try:
        classif_df = classif_df.drop("NR", axis=1)
    except:
        pass
    logging_header("Classification results")
    print(classif_df.to_string(index=False))
    classif_df.to_csv(arguments.output, index=False)
        
