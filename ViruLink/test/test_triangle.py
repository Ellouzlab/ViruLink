#!/usr/bin/env python3
"""
test_triangle.py – end-to-end trainer/evaluator for the OrdTri pipeline.

Key legacy behaviours restored:
    • Robust column remap for mmseqs ANI tables.
    • Node2Vec *always* runs (train_utils guarantees non-empty graph).
    • Relationship bounds built without k_classes clipping (old logic).
    • Parameter-name shim so n2v.py sees "window" instead of "window_size".
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ViruLink.default_yaml import default_yaml_dct
from ViruLink.setup.databases import database_info
from ViruLink.utils import logging_header
from ViruLink.train.n2v import n2v
from ViruLink.train.train_utils import (
    TriDS,
    fuse_embeddings,
    generate_score_profile_from_yaml,
    initiate_OrdTriTwoStageAttn,
    prepare_relationship_bounds,
    process_graph_for_n2v,
    remove_node_versions,
    run_epoch,
    sample_triangles_wrapper,
)

# ------------------------------------------------------------------ #
# Helper – legacy→new parameter key map for Node2Vec                 #
# ------------------------------------------------------------------ #
def _fix_n2v_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = cfg.copy()
    if "window" not in out and "window_size" in out:
        out["window"] = out["window_size"]
    return out


# ------------------------------------------------------------------ #
# Hierarchical metrics (unchanged)                                   #
# ------------------------------------------------------------------ #
def _hier_metrics(cm: np.ndarray | None, labels: List[str], nr_code: int) -> Dict[str, Any]:
    if cm is None or not labels:
        return {"per_class": {}, "spearman_rho": float("nan"), "kendall_tau": float("nan")}

    from scipy.stats import spearmanr, kendalltau

    arr = np.asarray(cm, dtype=int)
    out: Dict[str, Any] = {"per_class": {}}
    for i, lbl in enumerate(labels):
        if i == nr_code:
            tp = arr[i, i]
            pred = arr[:, i].sum()
            act = arr[i, :].sum()
        else:
            tp = arr[i:, i:].sum()
            pred = arr[:, i:].sum()
            act = arr[i:, :].sum()
        prec = tp / pred if pred else 0.0
        rec = tp / act if act else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["per_class"][lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": int(act)}

    tr, pr = [], []
    for i in range(len(labels)):
        sup = arr[i, :].sum()
        if sup:
            tr.append(i)
            pr.append((arr[i, :] * np.arange(len(labels))).sum() / sup)
    if len(tr) > 1:
        out["spearman_rho"] = spearmanr(tr, pr)[0]
        out["kendall_tau"] = kendalltau(tr, pr)[0]
    return out


# ------------------------------------------------------------------ #
# NamedTuple for prepare() output                                     #
# ------------------------------------------------------------------ #
class PrepOut(NamedTuple):
    k_classes: int
    nr_code: int
    comb_dim: int
    loaders: Dict[str, DataLoader]
    labels: List[str]


# ------------------------------------------------------------------ #
# Data-prep for a single database                                     #
# ------------------------------------------------------------------ #
def prepare_data_for_db(
    args: Any,
    db_name: str,
    trn_cfg: Dict[str, Any],
    mdl_cfg: Dict[str, Any],
    ani_cfg: Dict[str, Any],
    n2v_cfg_raw: Dict[str, Any],
    lvl_map: Dict[str, int],
) -> PrepOut:
    base = Path(args.databases_loc) / db_name

    k_classes = max(lvl_map.values()) + 1
    labels = sorted(lvl_map, key=lvl_map.get)
    nr_code = lvl_map.get("NR", -1)

    # load ANI / HYP edge tables
    ani_fn = "mmseqs_ANI.tsv" if ani_cfg.get("ani_program", "skani") == "mmseqs" else "self_ANI.tsv"
    ani_df = remove_node_versions(pd.read_csv(base / ani_fn, sep="\t"))
    
    # ensure mmseqs has the same column names as skani
    if ani_cfg.get("ani_program", "skani") == "mmseqs":
        colmap = {}
        for c in ("query", "source"):
            if c in ani_df.columns:
                colmap[c] = "source"
                break
        for c in ("target", "subject"):
            if c in ani_df.columns:
                colmap[c] = "target"
                break
        for c in ("ani", "ANI", "pident", "identity"):
            if c in ani_df.columns:
                colmap[c] = "weight"
                break
        ani_df = ani_df.rename(columns=colmap)

    hyp_df = remove_node_versions(pd.read_csv(base / "hypergeom_edges.csv"))

    # Rescale
    if mdl_cfg.get("rescale_ani_weights") and "weight" in ani_df:
        w = ani_df["weight"].to_numpy(dtype=float)
        rng = np.nanmax(w) - np.nanmin(w)
        ani_df["weight"] = (w - np.nanmin(w)) / rng if rng > 1e-9 else (1.0 if np.nanmax(w) > 0.5 else 0.0)

    # graphs & embeddings
    ani_graph = process_graph_for_n2v(ani_df)
    hyp_graph = process_graph_for_n2v(hyp_df)

    n2v_params = _fix_n2v_cfg(n2v_cfg_raw)
    emb_dim = n2v_params["embedding_dim"]
    ani_emb = n2v(ani_graph, int(args.threads), n2v_params)
    hyp_emb = n2v(hyp_graph, int(args.threads), n2v_params)
    fused = fuse_embeddings(ani_emb, hyp_emb, emb_dim)
    comb_dim = emb_dim * 2

    # splits
    nodes = list(fused)
    random.Random(trn_cfg["RNG_seed"]).shuffle(nodes)
    s1, s2 = int(0.8 * len(nodes)), int(0.9 * len(nodes))
    splits = {"train": nodes[:s1], "val": nodes[s1:s2], "test": nodes[s2:]}

    # relationship bounds & triangles 
    meta = pd.read_csv(base / f"{db_name}.csv")
    rel_bounds = prepare_relationship_bounds(meta, lvl_map)  # no k_classes arg

    tri_train = trn_cfg["TRIANGLES_PER_CLASS_train"]
    tri_eval = trn_cfg["TRIANGLES_PER_CLASS_eval"]
    loaders: Dict[str, DataLoader] = {}

    for sp, nodelist in splits.items():
        if not nodelist:
            continue
        sub_edges = rel_bounds[
            rel_bounds["source"].isin(nodelist) & rel_bounds["target"].isin(nodelist)
        ].reset_index(drop=True)
        tris = sample_triangles_wrapper(
            nodelist,
            sub_edges,
            tri_train if sp == "train" else tri_eval,
            k_classes,
            int(args.threads),
            trn_cfg["RNG_seed"] + {"train": 0, "val": 1, "test": 2}[sp],
        )
        ds = TriDS(tris, fused, ani_graph, hyp_graph, comb_dim)
        loaders[sp] = DataLoader(
            ds,
            trn_cfg["BATCH"],
            shuffle=(sp == "train"),
            num_workers=max(0, int(args.threads) // 2),
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )

    return PrepOut(k_classes, nr_code, comb_dim, loaders, labels)


# ------------------------------------------------------------------ #
# Train + evaluate one database                                      #
# ------------------------------------------------------------------ #
def run_db(
    args: Any,
    db: str,
    trn_cfg: Dict[str, Any],
    mdl_cfg: Dict[str, Any],
    ani_cfg: Dict[str, Any],
    n2v_cfg: Dict[str, Any],
    lvl_map: Dict[str, int],
):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = prepare_data_for_db(args, db, trn_cfg, mdl_cfg, ani_cfg, n2v_cfg, lvl_map)
    if not data.loaders:
        logging.error(f"{db}: no data – skipping.")
        return

    model = initiate_OrdTriTwoStageAttn(mdl_cfg, data.k_classes, data.comb_dim, device)
    opt = torch.optim.AdamW(model.parameters(), lr=trn_cfg["LEARNING_RATE"], weight_decay=1e-5)

    best, bad = float("inf"), 0
    for epoch in range(1, trn_cfg["EPOCHS"] + 1):
        tl, th, ta, _ = run_epoch(model, data.loaders["train"], data.k_classes, opt, device, mdl_cfg)
        vl = vh = va = float("nan")
        if "val" in data.loaders:
            vl, vh, va, _ = run_epoch(model, data.loaders["val"], data.k_classes, None, device, mdl_cfg)

        logging.info(
            f"{db}  Ep {epoch:02d}:  "
            f"T-loss={tl:.4f}  T-hit={th:.3f}  T-acc={ta:.3f} |  "
            f"V-loss={vl:.4f}  V-hit={vh:.3f}  V-acc={va:.3f}"
        )

        if not np.isnan(vl) and vl < best:
            best = vl
            bad = 0
            torch.save(model.state_dict(), f"{db}_best.pt")
        elif not np.isnan(vl):
            bad += 1
            if bad >= mdl_cfg.get("early_stopping_patience", 5):
                logging.info(f"{db}: early-stopping after {epoch} epochs.")
                break

    # reload best and evaluate
    if Path(f"{db}_best.pt").exists():
        model.load_state_dict(torch.load(f"{db}_best.pt", map_location=device))
    for sp, loader in data.loaders.items():
        l, h, a, cm = run_epoch(model, loader, data.k_classes, None, device, mdl_cfg, collect_cm=True)
        logging.info(f"{db} {sp.upper():5s}  loss={l:.4f}  hit={h:.3f}  acc={a:.3f}")
        if cm is not None:
            logging.info(json.dumps(_hier_metrics(cm, data.labels, data.nr_code)))


# ------------------------------------------------------------------ #
# Main handler                                                       #
# ------------------------------------------------------------------ #
def TestHandler(args: Any):
    cfg = default_yaml_dct
    mode = "normal"
    m = cfg["settings"][mode]
    trn_cfg = m["training_params"]
    mdl_cfg = trn_cfg["Model"]
    n2v_cfg = _fix_n2v_cfg(m["n2v"])
    ani_cfg = m["graph_making"]["ANI"]

    dbs = database_info()["Class"].unique().tolist()
    score_cfg = generate_score_profile_from_yaml(trn_cfg["Levels"], dbs)

    targets = dbs if args.all else [args.database]
    for db in targets:
        if db == "VOGDB":
            continue
        logging_header(f"Processing {db}")
        run_db(args, db, trn_cfg, mdl_cfg, ani_cfg, n2v_cfg, score_cfg[db])
        logging_header(f"Finished {db}")
