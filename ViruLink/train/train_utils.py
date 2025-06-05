"""
ViruLink – training utilities  (legacy-compatible full version)
================================================================
Highlights
----------
* Guarantees Node2Vec always sees at least one edge (self-loop fallback).
* `prepare_relationship_bounds()` works with or without `k_classes`.
* Exports everything test_triangle expects: TriDS, run_epoch,
  initiate_OrdTriTwoStageAttn, etc.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

from ViruLink.train.OrdTri import OrdTriTwoStageAttn
from ViruLink.train.losses import (
    CUM_TRE_LOSS,
    cum_interval_bce,
    exp_rank_cum,
    _adjacent_probs,
)
from ViruLink.sampler.triangle_sampler import sample_triangles
from ViruLink.relations.relationship_edges import build_relationship_edges


# -------------------------------------------------------------------- #
# 1. Generic helpers                                                   #
# -------------------------------------------------------------------- #
def remove_node_versions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ("source", "target"):
        if col in out.columns and out[col].dtype == "object":
            out[col] = out[col].str.split(".").str[0]
    return out


def process_graph_for_n2v(df: pd.DataFrame) -> pd.DataFrame:
    """Return undirected graph with self-loops; never empty."""
    if df is None or df.empty:
        logging.warning("Empty edge list – adding dummy self-loop.")
        return pd.DataFrame({"source": ["_dummy"], "target": ["_dummy"], "weight": [1.0]})

    dfp = df.copy()
    dfp["weight"] = pd.to_numeric(dfp.get("weight", np.nan), errors="coerce")
    dfp["source"] = dfp["source"].astype(str)
    dfp["target"] = dfp["target"].astype(str)

    core = dfp[dfp["source"] != dfp["target"]].copy()
    if core.empty:
        nodes = pd.unique(dfp[["source", "target"]].values.ravel("K"))
        mw = float(dfp["weight"].max(skipna=True))
        mw = mw if pd.notna(mw) else 0.0
        return pd.DataFrame({"source": nodes, "target": nodes, "weight": mw})

    u = np.minimum(core["source"], core["target"])
    v = np.maximum(core["source"], core["target"])
    und = (
        core.assign(u_tmp=u, v_tmp=v)
        .groupby(["u_tmp", "v_tmp"], as_index=False)["weight"]
        .max()
        .rename(columns={"u_tmp": "u", "v_tmp": "v"})
    )
    nodes = pd.unique(core[["source", "target"]].values.ravel("K"))
    max_w = float(und["weight"].max(skipna=True))

    loops = pd.DataFrame({"u": nodes, "v": nodes, "weight": max_w})
    g = pd.concat([und, und.rename(columns={"u": "v", "v": "u"}), loops])
    g = g.rename(columns={"u": "source", "v": "target"})
    if g.empty:
        return pd.DataFrame({"source": ["_dummy"], "target": ["_dummy"], "weight": [1.0]})
    return g


def fuse_embeddings(
    emb1: Dict[str, np.ndarray], emb2: Dict[str, np.ndarray], part_dim: int
) -> Dict[str, np.ndarray]:
    z = np.zeros(part_dim, dtype=np.float32)
    return {
        n: np.concatenate([emb1.get(n, z), emb2.get(n, z)])
        for n in set(emb1) | set(emb2)
    }


def generate_score_profile_from_yaml(
    yaml_levels: Dict[str, bool], dbs: List[str]
) -> Dict[str, Dict[str, int]]:
    active = [lvl for lvl, on in yaml_levels.items() if on]
    rank = {lvl: i for i, lvl in enumerate(active)}
    return {db: rank.copy() for db in dbs}


# -------------------------------------------------------------------- #
# 2. Relationship bounds                                               #
# -------------------------------------------------------------------- #
def prepare_relationship_bounds(
    meta_df: pd.DataFrame,
    lvl2rank: Dict[str, int],
    k_classes: Optional[int] = None,
) -> pd.DataFrame:
    if meta_df.empty or not lvl2rank:
        return pd.DataFrame(columns=["source", "target", "lower", "upper"])
    if "Accession" not in meta_df.columns:
        logging.error("metadata missing Accession column")
        return pd.DataFrame(columns=["source", "target", "lower", "upper"])

    if k_classes is None:
        edges = build_relationship_edges(meta_df, lvl2rank)
    else:
        edges = build_relationship_edges(meta_df, lvl2rank, k_classes)

    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "lower", "upper"])
    return (
        edges.rename(columns={"rank_low": "lower", "rank_up": "upper"})
        .astype({"lower": "uint8", "upper": "uint8"})
        [["source", "target", "lower", "upper"]]
    )


# -------------------------------------------------------------------- #
# 3. Triangle-sampler wrapper                                          #
# -------------------------------------------------------------------- #
def sample_triangles_wrapper(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_cls: int,
    k_classes: int,
    threads: int,
    seed: int,
) -> List[Tuple]:
    if not nodes or rel_df.empty:
        return []
    try:
        return sample_triangles(
            [str(n) for n in nodes],
            rel_df["source"].astype(str).tolist(),
            rel_df["target"].astype(str).tolist(),
            rel_df["lower"].astype("uint8").tolist(),
            rel_df["upper"].astype("uint8").tolist(),
            int(num_per_cls),
            int(k_classes),
            int(threads),
            int(seed),
        )
    except Exception as e:
        logging.error("sample_triangles() failed", exc_info=e)
        return []


# -------------------------------------------------------------------- #
# 4. PyTorch Dataset                                                   #
# -------------------------------------------------------------------- #
class TriDS(Dataset):
    def __init__(
        self,
        tris: List[Tuple],
        emb: Dict[str, np.ndarray],
        ani_df: pd.DataFrame,
        hyp_df: pd.DataFrame,
        comb_dim: int,
    ):
        self.tris = tris
        self.emb = emb
        self.comb_dim = comb_dim
        self.ani_lut = self._lut(ani_df)
        self.hyp_lut = self._lut(hyp_df)

    @staticmethod
    def _lut(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        lut: Dict[Tuple[str, str], float] = {}
        if df is None or df.empty:
            return lut
        for s, t, w in df[["source", "target", "weight"]].itertuples(False, None):
            try:
                w = float(w)
            except Exception:
                w = 0.0
            lut[(str(s), str(t))] = w
            lut[(str(t), str(s))] = w
        return lut

    def __len__(self):
        return len(self.tris)

    def __getitem__(self, idx: int):
        q, r1, r2, b_qr1, b_qr2, b_r1r2 = self.tris[idx]
        zeros = np.zeros(self.comb_dim, np.float32)
        eq = torch.tensor(self.emb.get(str(q), zeros))
        ea = torch.tensor(self.emb.get(str(r1), zeros))
        eh = torch.tensor(self.emb.get(str(r2), zeros))

        def w(a, b, lut):
            return torch.tensor([lut.get((str(a), str(b)), 0.0)], dtype=torch.float32)

        edge = torch.cat(
            [
                w(q, r1, self.ani_lut),
                w(q, r1, self.hyp_lut),
                w(q, r2, self.ani_lut),
                w(q, r2, self.hyp_lut),
                w(r1, r2, self.ani_lut),
                w(r1, r2, self.hyp_lut),
            ]
        )
        return {
            "eq": eq,
            "ea": ea,
            "eh": eh,
            "edge": edge,
            "lqa": torch.tensor(b_qr1, dtype=torch.long),
            "lqh": torch.tensor(b_qr2, dtype=torch.long),
            "lrr": torch.tensor(b_r1r2, dtype=torch.long),
        }


# -------------------------------------------------------------------- #
# 5. Model factory                                                     #
# -------------------------------------------------------------------- #
def initiate_OrdTriTwoStageAttn(
    model_params: Dict[str, Any],
    k_classes: int,
    n2v_embedding_dim: int,
    device: torch.device,
) -> OrdTriTwoStageAttn:
    logging.info(
        f"Init OrdTriTwoStageAttn  k={k_classes}  dim={n2v_embedding_dim}  params={model_params}"
    )
    needed = [
        "Activation",
        "edge_feature_dim_refiner",
        "attn_heads",
        "attn_layers",
        "attn_dropout",
    ]
    for k in needed:
        if k not in model_params:
            raise ValueError(f"Missing model param '{k}'")
    model = OrdTriTwoStageAttn(
        node_emb_dim=n2v_embedding_dim,
        k_classes=k_classes,
        act=model_params["Activation"],
        edge_feature_dim=model_params["edge_feature_dim_refiner"],
        attn_heads=model_params["attn_heads"],
        attn_layers=model_params["attn_layers"],
        attn_dropout=model_params["attn_dropout"],
    )
    return model.to(device)


# -------------------------------------------------------------------- #
# 6. Training-/evaluation epoch                                        #
# -------------------------------------------------------------------- #
def interval_hit_rate(logits: torch.Tensor, bounds: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    pred = torch.argmax(_adjacent_probs(logits), dim=1)
    hits = (pred >= bounds[:, 0]) & (pred <= bounds[:, 1])
    return hits.float().mean().item()


def run_epoch(
    model: OrdTriTwoStageAttn,
    loader: DataLoader,
    k_classes: int,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    op_params: Dict[str, Any],
    collect_cm: bool = False,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    if loader is None or len(loader) == 0:
        return float("nan"), float("nan"), float("nan"), None

    model.train(opt is not None)
    max_recycles = op_params.get("max_recycles", 1)
    lam_int = op_params.get("lambda_int", 0.1)
    lam_tri = op_params.get("lambda_tri", 0.1)
    lam_aux = op_params.get("aux_loss_weight", 0.1)
    lam_mono = op_params.get("mono_lambda", 0.0)
    gate = op_params.get("gate_alpha", 0.1)

    tot_loss = tot_hit = tot_acc_pt = 0.0
    n_samp = n_pt = 0
    true_lab: List[int] = []
    pred_lab: List[int] = []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        bs = batch["eq"].size(0)
        passes = (
            random.randint(1, max_recycles) if opt is not None else max_recycles
        )

        rec = torch.full((bs, 3 * k_classes), 1.0 / k_classes, device=device)
        prev_er = None
        losses = []

        for p in range(passes):
            la, lh, lr, ila, ilh, ilr = model(
                batch["eq"], batch["ea"], batch["eh"], batch["edge"], rec
            )
            ploss = CUM_TRE_LOSS(la, lh, lr, batch, lam_int, lam_tri)
            if lam_aux > 0:
                ploss += lam_aux * (
                    cum_interval_bce(ila, batch["lqa"])
                    + cum_interval_bce(ilh, batch["lqh"])
                    + cum_interval_bce(ilr, batch["lrr"])
                )
            if lam_mono > 0:
                er = exp_rank_cum(la, k_classes)
                if prev_er is not None:
                    ploss += lam_mono * F.relu(prev_er - er).pow(2).mean()
                prev_er = er.detach()
            losses.append(ploss)

            if p < passes - 1:
                with torch.no_grad():
                    rec_la = _adjacent_probs(la.detach())
                    rec_lh = _adjacent_probs(lh.detach())
                    rec_lr = _adjacent_probs(lr.detach())
                    rec_new = torch.cat([rec_la, rec_lh, rec_lr], 1)
                    rec = gate * rec_new + (1.0 - gate) * rec

        loss = torch.stack(losses).mean()
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        tot_loss += loss.item() * bs
        n_samp += bs

        hit = interval_hit_rate(la, batch["lqa"])
        tot_hit += hit * bs

        # exact-point accuracy
        point_mask = batch["lqa"][:, 0] == batch["lqa"][:, 1]
        if point_mask.any():
            n_point = int(point_mask.sum())
            n_pt += n_point
            pred_pt = torch.argmax(_adjacent_probs(la[point_mask]), 1)
            true_pt = batch["lqa"][point_mask, 0]
            tot_acc_pt += (pred_pt == true_pt).float().sum().item()
            if collect_cm:
                true_lab.extend(true_pt.cpu().tolist())
                pred_lab.extend(pred_pt.cpu().tolist())

    avg_loss = tot_loss / n_samp if n_samp else float("nan")
    avg_hit = tot_hit / n_samp if n_samp else float("nan")
    avg_acc_pt = tot_acc_pt / n_pt if n_pt else float("nan")

    cm = (
        confusion_matrix(true_lab, pred_lab, labels=list(range(k_classes)))
        if collect_cm and true_lab
        else None
    )
    return avg_loss, avg_hit, avg_acc_pt, cm
