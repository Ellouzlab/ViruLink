#!/usr/bin/env python3
"""
Ordinal-relationship prediction with interval-censored losses
Triangle sampler + Node2Vec embeddings + raw ANI/HYP edge features.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ViruLink imports
from ViruLink.setup.score_profile import score_config
from ViruLink.setup.run_parameters import parameters
from ViruLink.setup.databases import database_info
from ViruLink.relations.relationship_edges import build_relationship_edges
from ViruLink.train.OrdTri import OrdTri
from ViruLink.train.n2v import n2v
from ViruLink.train.losses import CUM_TRE_LOSS, _adjacent_probs, exp_rank
from ViruLink.sampler.triangle_sampler import sample_triangles
from ViruLink.utils import logging_header

# hyper-parameters
RNG_SEED = 42
EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-3
NUM_PER_CLASS = 4_000  # triangles per (rank × upper|lower)
LAMBDA_INT = 1.0
LAMBDA_TRI = 0.3
EDGE_ORDER = ("r1r2", "qr2", "qr1")   # prediction sequence
EDGE_PRED_COUNT = 4

EMBED_DIM = parameters["embedding_dim"]
COMB_DIM = EMBED_DIM * 2          # after fusing ANI+HYP embeddings

EVALUATION_METRICS_ENABLED = True

# Device and RNG

torch.manual_seed(RNG_SEED)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# utility helpers
def remove_version(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("source", "target"):
        df[col] = df[col].str.split(".").str[0]
    return df


def process_graph(df: pd.DataFrame) -> pd.DataFrame:
    """Return an undirected, self-loop-completed graph."""
    df = df[df["source"] != df["target"]].copy()
    u = np.minimum(df["source"], df["target"])
    v = np.maximum(df["source"], df["target"])
    df[["u", "v"]] = np.column_stack([u, v])
    und = df.groupby(["u", "v"], as_index=False)["weight"].max()
    max_w = df["weight"].max()
    nodes = pd.unique(df[["source", "target"]].values.ravel())
    self_loops = pd.DataFrame({"u": nodes, "v": nodes, "weight": max_w})
    rev = und.rename(columns={"u": "v", "v": "u"})
    return (
        pd.concat([und, rev, self_loops], ignore_index=True)
        .rename(columns={"u": "source", "v": "target"})
    )


def fuse_emb(ani_emb: Dict[str, np.ndarray],
             hyp_emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    fused: Dict[str, np.ndarray] = {}
    for n in set(ani_emb) | set(hyp_emb):
        v1 = ani_emb.get(n, np.zeros(EMBED_DIM))
        v2 = hyp_emb.get(n, np.zeros(EMBED_DIM))
        fused[n] = np.concatenate([v1, v2])
    return fused


def build_rel_bounds(meta_df: pd.DataFrame, rel_scores: Dict[str, int]) -> pd.DataFrame:
    df = build_relationship_edges(meta_df, rel_scores)
    return (
        df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
        .astype({"lower": "uint8", "upper": "uint8"})
        [["source", "target", "lower", "upper"]]
    )


# triangle sampler wrapper

def sample_intra_split_triangles(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_class: int,
    k_classes: int,
    threads: int,
    seed: int = RNG_SEED,
) -> List[Tuple]:
    """call the cpp triangle sampler."""
    return sample_triangles(
        nodes,
        rel_df["source"].tolist(),
        rel_df["target"].tolist(),
        rel_df["lower"].astype("uint8").tolist(),
        rel_df["upper"].astype("uint8").tolist(),
        num_per_class,
        k_classes,
        threads,
        seed,
    )


# PyTorch dataset
class TriDS(Dataset):
    def __init__(
        self,
        tris: List[Tuple],
        emb: Dict[str, np.ndarray],
        ani_df: pd.DataFrame,
        hyp_df: pd.DataFrame,
    ) -> None:
        self.t = tris
        self.e = emb

        def lut(df: pd.DataFrame):
            d: Dict[Tuple[str, str], float] = {}
            for s, t, w in df[["source", "target", "weight"]].itertuples(False):
                d[(s, t)] = w; d[(t, s)] = w
            return d

        self.ani = lut(ani_df)
        self.hyp = lut(hyp_df)

    def _w(self, a: str, b: str, table: Dict[Tuple[str, str], float]):
        return torch.tensor([table.get((a, b), 0.0)], dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        q, r1, r2, b1, b2, b3 = self.t[i]
        eq = torch.tensor(self.e[q], dtype=torch.float32)
        ea = torch.tensor(self.e[r1], dtype=torch.float32)
        eh = torch.tensor(self.e[r2], dtype=torch.float32)

        edge = torch.cat(
            [
                self._w(q, r1, self.ani), self._w(q, r1, self.hyp),
                self._w(q, r2, self.ani), self._w(q, r2, self.hyp),
                self._w(r1, r2, self.ani), self._w(r1, r2, self.hyp),
            ]
        )  # → [6]

        return {
            "eq": eq,
            "ea": ea,
            "eh": eh,
            "edge": edge,
            "lqa": torch.tensor(b1, dtype=torch.long),
            "lqh": torch.tensor(b2, dtype=torch.long),
            "lrr": torch.tensor(b3, dtype=torch.long),
        }


# training/val helpers

def interval_hit_rate(logits: torch.Tensor, bounds: torch.Tensor) -> float:
    pred = torch.argmax(_adjacent_probs(logits), dim=1)
    lo, up = bounds[:, 0], bounds[:, 1]
    return ((pred >= lo) & (pred <= up)).float().mean().item()


def run_epoch(
    model: OrdTri,
    loader: DataLoader,
    k_classes: int,
    nr_code: int,
    opt: torch.optim.Optimizer | None = None,
    collect_cm: bool = False,
    cpu_flag: bool = False,
):
    """
    Multipass training loop inspired by AlphaFold recycling.

    - 1-to-3 passes per batch, chosen uniformly at random  
    - probabilities are detached between passes to stop gradients  
    - small residual-gate (0.12) mixes new ↔ old probs  
    - interval + triangle loss applied on **every** pass  
    - optional monotonic penalty nudges each pass to improve
    """
    MAX_RECYCLES = 3          # num passes
    GATE_ALPHA   = 0.12       # σ(-2) ≈ 0.12  => small corrections early on
    MONO_LAMBDA  = 0.10       # weight of the monotonic regulariser
    

    device_local = torch.device("cuda" if torch.cuda.is_available() and not cpu_flag else "cpu")
    totL = totH = 0.0
    totA = 0.0; n_pt = 0; n = 0

    tbuf, pbuf = [], []
    model.train(opt is not None)

    for batch in loader:
        # move to device
        for k in batch:
            batch[k] = batch[k].to(device_local)
        B = batch["eq"].size(0)

        # choose # of passes for this batch
        n_passes = random.randint(1, MAX_RECYCLES)

        # running edge-probabilities to recycle between passes
        uniform = torch.full((B, k_classes), 1.0 / k_classes, device=device_local)
        cur_p = {"qr1": uniform.clone(), "qr2": uniform.clone(), "r1r2": uniform.clone()}

        pass_losses = []
        prev_rank = None                      # for the monotonic penalty
        last_logits = {}                      # to keep the *final* pass’s logits

        # outer recycle loop
        for _pass in range(n_passes):
            # make a local copy
            local_p = {k: v.clone() for k, v in cur_p.items()}

            # one internal refinement cycle
            for _ in range(EDGE_PRED_COUNT):
                for edge_key in EDGE_ORDER:
                    if edge_key == "qr1":
                        xa, xb = batch["eq"], batch["ea"]
                    elif edge_key == "qr2":
                        xa, xb = batch["eq"], batch["eh"]
                    else:
                        xa, xb = batch["ea"], batch["eh"]

                    probs_vec = torch.cat(
                        [local_p["qr1"], local_p["qr2"], local_p["r1r2"]], dim=1
                    )
                    lg = model(xa, xb, batch["edge"], probs_vec)
                    last_logits[edge_key] = lg
                    local_p[edge_key] = _adjacent_probs(lg)

            # loss for this pass
            la = last_logits["qr1"]; lh = last_logits["qr2"]; lr = last_logits["r1r2"]
            L = CUM_TRE_LOSS(la, lh, lr, batch)

            # monotonic don’t get worse penalty
            rank_now = exp_rank(la, k_classes)
            if prev_rank is not None:
                mono = F.relu(prev_rank - rank_now).pow(2).mean()
                L = L + MONO_LAMBDA * mono
            prev_rank = rank_now.detach()

            pass_losses.append(L)

            # recycle probs with a residual gate
            for k in cur_p:
                cur_p[k] = (GATE_ALPHA * local_p[k].detach() +
                            (1.0 - GATE_ALPHA) * cur_p[k])

        # end recycle loop

        # average loss over the passes
        loss = torch.stack(pass_losses).mean()

        # optimise
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # metrics on the final pass
        hit = interval_hit_rate(last_logits["qr1"], batch["lqa"])
        mask = batch["lqa"][:, 0] == batch["lqa"][:, 1]
        p_cnt = mask.sum().item()
        if p_cnt:
            preds = torch.argmax(_adjacent_probs(last_logits["qr1"][mask]), 1)
            acc = (preds == batch["lqa"][mask, 0]).float().mean().item()
            totA += acc * p_cnt; n_pt += p_cnt
            if collect_cm:
                tbuf.extend(batch["lqa"][mask, 0].cpu().tolist())
                pbuf.extend(preds.cpu().tolist())

        totL += loss.item() * B; totH += hit * B; n += B

    # confusion-matrix + summary stats
    cm = (
        confusion_matrix(tbuf, pbuf, labels=list(range(k_classes))) if tbuf else None
    )
    meanA = totA / n_pt if n_pt else float("nan")
    return totL / n, totH / n, meanA, cm


def compute_eval_cm_metrics(cm: np.ndarray, labels: List[str], nr_code: int):
    from scipy.stats import spearmanr, kendalltau

    arr = np.array(cm, int); K = len(labels)
    metrics = {"per_class": {}}
    for r, lbl in enumerate(labels):
        if r == nr_code:
            TP = arr[r, r]; P = arr[:, r].sum(); A = arr[r, :].sum()
        else:
            TP = arr[r:, r:].sum(); P = arr[:, r:].sum(); A = arr[r:, :].sum()
        prec = TP / P if P else 0.0; rec = TP / A if A else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        metrics["per_class"][lbl] = {
            "precision": prec, "recall": rec, "f1": f1, "support": int(A)
        }
    tr, pred = [], []
    for i in range(K):
        sup = arr[i, :].sum();
        if not sup: continue
        tr.append(i)
        pred.append((arr[i, :] * np.arange(K)).sum() / sup)
    if len(tr) > 1:
        rho, _ = spearmanr(tr, pred); tau, _ = kendalltau(tr, pred)
    else:
        rho = tau = float("nan")
    metrics["spearman_rho"] = rho; metrics["kendall_tau"] = tau
    return metrics

# main routine

def DatabaseTesting(args, db: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # per-database rank helpers
    lvl2rank = {lvl: r for r, lvl in enumerate(score_config[db])}
    k_classes = max(lvl2rank.values()) + 1
    nr_code = lvl2rank["NR"]
    RESCALE_ANI = False
    act = "relu" if not args.swiglu else "swiglu"

    # Patch the global symbols
    global K_CLASSES, NR_CODE
    K_CLASSES = k_classes
    NR_CODE = nr_code

    p = Path(args.databases_loc) / db

    logging_header("Loading %s database", db)
    ani_edges = remove_version(pd.read_csv(p / "self_ANI.tsv", sep="\t"))
    hyp_edges = remove_version(pd.read_csv(p / "hypergeom_edges.csv"))

    if RESCALE_ANI:
        w = ani_edges["weight"].to_numpy(dtype=float)
        rng = w.max() - w.min()
        if rng:
            ani_edges["weight"] = (w - w.min()) / rng
        else:
            ani_edges["weight"] = 1.0

    ani = process_graph(ani_edges)
    hyp = process_graph(hyp_edges)

    logging.info("ANI Graph: %d edges", len(ani))
    logging.info("HYP Graph: %d edges", len(hyp))

    logging_header("Performing node2vec")
    ani_emb = n2v(ani, args.threads, parameters)
    hyp_emb = n2v(hyp, args.threads, parameters)
    emb = fuse_emb(ani_emb, hyp_emb)
    logging.info("Node2vec embeddings generated for %d nodes", len(emb))

    all_nodes = list(emb)
    random.shuffle(all_nodes)
    c1, c2 = int(.8 * len(all_nodes)), int(.9 * len(all_nodes))
    splits = {"train": all_nodes[:c1], "val": all_nodes[c1:c2], "test": all_nodes[c2:]}
    logging.info("train/val/test sizes: %d/%d/%d", *(len(splits[k]) for k in ("train", "val", "test")))

    logging_header("Loading %s relationships", db)
    meta = pd.read_csv(p / f"{db}.csv")
    rel = build_rel_bounds(meta, score_config[db])
    edges = {
        k: rel[rel["source"].isin(splits[k]) & rel["target"].isin(splits[k])].reset_index(drop=True)
        for k in splits
    }
    logging.info("Relationship edges: train - %d | val - %d | test - %d",
                 len(edges["train"]), len(edges["val"]), len(edges["test"]))

    logging_header("Sampling Triangles")
    logging.info("This may take a while …")
    tris = {
        k: sample_intra_split_triangles(
            splits[k], edges[k],
            NUM_PER_CLASS if k == "train" else NUM_PER_CLASS // 8,
            k_classes, args.threads, RNG_SEED,
        ) for k in splits
    }
    for k in tris:
        logging.info("%s triangles = %d", k, len(tris[k]))

    ds = {k: TriDS(tris[k], emb, ani, hyp) for k in tris}
    ld = {k: DataLoader(ds[k], BATCH_SIZE, shuffle=(k == "train")) for k in tris}

    model = OrdTri(COMB_DIM, k_classes, act=act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    logging_header("Training Edge Predictor")
    logging.info(f"Utilizing {device} for training!")

    best_val_loss = float("inf")
    best_model_state = None

    for ep in range(1, EPOCHS + 1):
        trL, trH, trA, _ = run_epoch(model, ld["train"], k_classes, nr_code, opt, cpu_flag=args.cpu)
        vaL, vaH, vaA, _ = run_epoch(model, ld["val"], k_classes, nr_code, cpu_flag=args.cpu)

        if vaL < best_val_loss:
            best_val_loss = vaL
            best_model_state = model.state_dict()

        logging.info("Ep%02d  train L=%.3f hit=%.3f acc=%.3f   val L=%.3f hit=%.3f acc=%.3f",
                     ep, trL, trH, trA, vaL, vaH, vaA)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Restored best model with val loss = %.4f", best_val_loss)

    # Final evaluation
    results = {
        k: run_epoch(model, ld[k], k_classes, nr_code, collect_cm=True, cpu_flag=args.cpu)
        for k in ("train", "val", "test")
    }
    levels = list(score_config[db])[:k_classes]

    for split, (_, _, _, CM) in results.items():
        if CM is None:
            logging.info("%s — no point-label edges —", split.upper())
        else:
            logging.info("%s confusion matrix:\n%s",
                         split.upper(), pd.DataFrame(CM, index=levels, columns=levels).to_string())

    logging.info("\n%s  hit=%0.3f/%0.3f/%0.3f  acc=%0.3f/%0.3f/%0.3f",
                 db, results['train'][1], results['val'][1], results['test'][1],
                 results['train'][2], results['val'][2], results['test'][2])

    if EVALUATION_METRICS_ENABLED:
        for split in ("train", "val", "test"):
            CM = results[split][3]
            if CM is None:
                continue
            extra = compute_eval_cm_metrics(CM, levels, nr_code)
            logging.info("%s hierarchical metrics:", split.upper())
            for lvl, m in extra["per_class"].items():
                logging.info("  %-6s prec=%0.3f rec=%0.3f f1=%0.3f sup=%d",
                             lvl, m["precision"], m["recall"], m["f1"], m["support"])
            logging.info("  Spearman rho=%0.3f, Kendall tau=%0.3f",
                         extra["spearman_rho"], extra["kendall_tau"])

    logging_header("Finished %s database", db)


# CLI entry point
def TestHandler(args):
    dbs = database_info()["Class"] if args.all else [args.database]
    for db in dbs:
        if db == "VOGDB":
            continue
        DatabaseTesting(args, db)
