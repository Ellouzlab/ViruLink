#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Ordinal-relationship prediction with interval-censored losses
# Vectorized pure-Python triangle sampler:
#   • stratified by rank (upper / lower) inside each split
#   • guarantees NUM_PER_CLASS × K_CLASSES × 2 triangles per split
#   • no cross-split leakage
# --------------------------------------------------------------------------
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ---------- ViruLink utilities ------------------------------------------
from ViruLink.setup.score_profile  import score_config
from ViruLink.setup.run_parameters import parameters
from ViruLink.setup.databases      import database_info
from ViruLink.utils import (
    prepare_edges_for_cpp,
    make_all_nodes_list,
    run_biased_random_walk,
)
from ViruLink.relations.relationship_edges import build_relationship_edges

# ---------- hyper-parameters --------------------------------------------
RNG_SEED      = 42
EPOCHS        = 20
BATCH_SIZE    = 512
LR            = 1e-3

NUM_PER_CLASS = 4000  # per rank × (upper | lower)

LAMBDA_INT    = 1.0
LAMBDA_TRI    = 0.3

EMBED_DIM     = parameters["embedding_dim"]
COMB_DIM      = EMBED_DIM * 2

# Toggle evaluation metrics on/off
EVALUATION_METRICS_ENABLED = True

# build our mapping level → rank
LEVEL2RANK = {
    lvl: r
    for r, lvl in enumerate(score_config["Caudoviricetes"].keys())
}
K_CLASSES = max(LEVEL2RANK.values()) + 1
NR_CODE   = LEVEL2RANK["NR"]   # index of the lowest ("NR") rank

# invert for labeling
RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RNG_SEED)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ======================================================================
# 1.  Node2Vec helpers
# ======================================================================
def run_walk(df: pd.DataFrame, thr: int):
    r, c, wf, l2id, id2l = prepare_edges_for_cpp(
        df["source"], df["target"], df["weight"]
    )
    walks = run_biased_random_walk(
        r, c, wf,
        make_all_nodes_list(l2id),
        parameters["walk_length"],
        parameters["p"], parameters["q"],
        thr, parameters["walks_per_node"]
    )
    return walks, id2l


def word2vec_emb(walks, id2lbl, thr: int):
    from gensim.models import Word2Vec
    model = Word2Vec(
        [[str(n) for n in w] for w in walks],
        vector_size=EMBED_DIM,
        window=parameters["window"],
        min_count=0,
        sg=1,
        workers=thr,
        epochs=parameters["epochs"]
    )
    zeros = np.zeros(EMBED_DIM, dtype=np.float32)
    return {
        lbl: (model.wv[str(nid)] if str(nid) in model.wv else zeros)
        for nid, lbl in id2lbl.items()
    }


def remove_version(df: pd.DataFrame):
    for col in ("source", "target"):
        df[col] = df[col].str.split(".").str[0]
    return df


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
    return pd.concat([und, rev, self_loops], ignore_index=True).rename(
        columns={"u": "source", "v": "target"}
    )


def fuse_emb(ani_emb: Dict[str, np.ndarray],
             hyp_emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    d = EMBED_DIM
    fused = {}
    for n in set(ani_emb) | set(hyp_emb):
        v1, v2 = ani_emb.get(n), hyp_emb.get(n)
        if v1 is None and v2 is None:
            continue
        if v1 is None: v1 = np.zeros(d)
        if v2 is None: v2 = np.zeros(d)
        fused[n] = np.concatenate([v1, v2])
    return fused


def build_embeddings(ani_df: pd.DataFrame,
                     hyp_df: pd.DataFrame,
                     thr: int = 8):
    ani_w, id2a = run_walk(ani_df, thr)
    hyp_w, id2h = run_walk(hyp_df, thr)
    ani_emb = word2vec_emb(ani_w, id2a, thr)
    hyp_emb = word2vec_emb(hyp_w, id2h, thr)
    return fuse_emb(ani_emb, hyp_emb)


# ======================================================================
# 2.  taxonomy helper
# ======================================================================
def build_rel_bounds(meta_df: pd.DataFrame,
                     rel_scores: Dict[str, int]) -> pd.DataFrame:
    df = build_relationship_edges(meta_df, rel_scores)
    return (
        df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
          .astype({"lower": "uint8", "upper": "uint8"})
          [["source", "target", "lower", "upper"]]
    )


# ======================================================================
# 3.  vectorized pure-Python sampler
# ======================================================================
def sample_intra_split_triangles(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_class: int,
    k_classes: int,
    rng: np.random.Generator
) -> List[Tuple]:
    rel = rel_df[["source", "target", "lower", "upper"]]
    u = np.minimum(rel["source"], rel["target"])
    v = np.maximum(rel["source"], rel["target"])
    key = u + "|" + v
    lut = dict(zip(
        key,
        rel[["lower", "upper"]]
           .itertuples(index=False, name=None)
    ))

    uppers = {r: [] for r in range(k_classes)}
    lowers = {r: [] for r in range(k_classes)}
    for s, t, lo, up in rel.itertuples(index=False, name=None):
        uppers[up].append((s, t))
        lowers[lo].append((s, t))

    primaries = []
    def pick(bucket):
        idx = rng.integers(0, len(bucket), num_per_class, dtype=np.int64)
        return [bucket[i] for i in idx]

    for r in range(k_classes):
        if not uppers[r] or not lowers[r]:
            raise RuntimeError(f"rank {r}: empty bucket")
        primaries.extend(pick(uppers[r]))
        primaries.extend(pick(lowers[r]))

    dfp = pd.DataFrame(primaries, columns=["q", "r1"])
    mask = rng.random(size=len(dfp)) < 0.5
    dfp.loc[mask, ["q", "r1"]] = dfp.loc[mask, ["r1", "q"]].values

    arr_nodes = np.array(nodes, dtype="U")
    dfp["r2"] = rng.choice(arr_nodes, size=len(dfp))
    bad = (dfp["r2"] == dfp["q"]) | (dfp["r2"] == dfp["r1"])
    while bad.any():
        dfp.loc[bad, "r2"] = rng.choice(arr_nodes, size=bad.sum())
        bad = (dfp["r2"] == dfp["q"]) | (dfp["r2"] == dfp["r1"])

    def add_bounds(a, b, lo_col, up_col):
        u2 = np.minimum(dfp[a], dfp[b])
        v2 = np.maximum(dfp[a], dfp[b])
        k2 = u2 + "|" + v2
        lo, up = zip(*(lut.get(x, (NR_CODE, NR_CODE)) for x in k2))
        dfp[lo_col] = lo
        dfp[up_col] = up

    add_bounds("q", "r1", "b1_lo", "b1_up")
    add_bounds("q", "r2", "b2_lo", "b2_up")
    add_bounds("r1", "r2", "b3_lo", "b3_up")

    return list(zip(
        dfp["q"], dfp["r1"], dfp["r2"],
        zip(dfp["b1_lo"], dfp["b1_up"]),
        zip(dfp["b2_lo"], dfp["b2_up"]),
        zip(dfp["b3_lo"], dfp["b3_up"])
    ))


# ======================================================================
# 4.  dataset, model, losses, metrics
# ======================================================================
class TriDS(Dataset):
    def __init__(self, tris, emb):
        self.t = tris
        self.e = emb

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        q, r1, r2, b1, b2, b3 = self.t[i]
        return {
            "eq": torch.tensor(self.e[q], dtype=torch.float32),
            "ea": torch.tensor(self.e[r1], dtype=torch.float32),
            "eh": torch.tensor(self.e[r2], dtype=torch.float32),
            "lqa": torch.tensor(b1, dtype=torch.long),
            "lqh": torch.tensor(b2, dtype=torch.long),
            "lrr": torch.tensor(b3, dtype=torch.long),
        }


class OrdTri(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim * 3 + 2, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
        )
        self.h1 = nn.Linear(64, k)
        self.h2 = nn.Linear(64, k)

    def _feat(self, q, r, brr):
        return torch.cat([q, r, q - r, brr.float() / (K_CLASSES - 1)], dim=-1)

    def forward(self, b):
        f1 = self.base(self._feat(b["eq"], b["ea"], b["lrr"]))
        f2 = self.base(self._feat(b["eq"], b["eh"], b["lrr"]))
        return self.h1(f1), self.h2(f2)


def interval_ce(logits, bounds):
    p = F.softmax(logits, dim=1)
    m = torch.zeros_like(p)
    for i, (lo, up) in enumerate(bounds.tolist()):
        m[i, lo : up + 1] = 1.0
    return -(torch.clamp((p * m).sum(1), min=1e-12).log()).mean()


def exp_rank(logits):
    return (F.softmax(logits, dim=1) * torch.arange(K_CLASSES, device=logits.device)).sum(1)


def hinge_sq(pred, lo, up):
    return (torch.relu(lo - pred) ** 2 + torch.relu(pred - up) ** 2)


def tri_dual(r1, r2, lb, ub):
    l1 = hinge_sq(r1, torch.minimum(r2, lb.float()), torch.minimum(r2, ub.float()))
    l2 = hinge_sq(r2, torch.minimum(r1, lb.float()), torch.minimum(r1, ub.float()))
    return (l1 + l2).mean()


def full_loss(la, lh, b):
    ce  = interval_ce(la, b["lqa"]) + interval_ce(lh, b["lqh"])
    tri = tri_dual(exp_rank(la), exp_rank(lh), b["lrr"][:, 0], b["lrr"][:, 1])
    return LAMBDA_INT * ce + LAMBDA_TRI * tri


def interval_hit_rate(logits, bounds):
    probs = F.softmax(logits, dim=1)
    mask  = torch.zeros_like(probs)
    for i, (lo, up) in enumerate(bounds.tolist()):
        mask[i, lo : up + 1] = 1.0
    return (probs * mask).sum(1).mean().item()


def run_epoch(model, loader, opt=None, collect_cm=False):
    totL = totH = totA = n = 0.0
    tbuf, pbuf = [], []
    training = opt is not None
    model.train(training)

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        la, lh = model(batch)
        loss = full_loss(la, lh, batch)
        if training:
            opt.zero_grad()
            loss.backward()
            opt.step()

        hit = interval_hit_rate(la, batch["lqa"])
        mask = batch["lqa"][:, 0] == batch["lqa"][:, 1]
        if mask.any():
            acc = (torch.argmax(la[mask], 1) == batch["lqa"][mask, 0]).float().mean().item()
        else:
            acc = float("nan")

        bs = batch["eq"].size(0)
        totL += loss.item() * bs
        totH += hit * bs
        totA += acc * bs if not np.isnan(acc) else 0
        n    += bs

        if collect_cm and mask.any():
            tbuf.extend(batch["lqa"][mask, 0].cpu().tolist())
            pbuf.extend(torch.argmax(la[mask], 1).cpu().tolist())

    mean_L   = totL / n
    mean_hit = totH / n
    mean_acc = totA / n
    cm = confusion_matrix(tbuf, pbuf, labels=list(range(K_CLASSES))) if tbuf else None
    return mean_L, mean_hit, mean_acc, cm


def compute_eval_cm_metrics(cm, labels, run_rank_correlation=True):
    """
    Compute hierarchical precision/recall/F1 for all ordinal thresholds,
    but for the lowest NR_CODE rank, use exact-match precision/recall.
    Optionally compute Spearman’s rho and Kendall’s tau.
    """
    import numpy as np
    from scipy.stats import spearmanr, kendalltau

    cm_arr = np.array(cm, dtype=int)
    K = len(labels)
    metrics = {'per_class': {}}

    for r in range(K):
        lbl = labels[r]
        if r == NR_CODE:
            # for the true 'NR' rank, only exact matches count
            TP = cm_arr[r, r]
            P  = cm_arr[:, r].sum()
            A  = cm_arr[r, :].sum()
        else:
            # hierarchical: any true ≥ r and pred ≥ r
            TP = cm_arr[r:, r:].sum()
            P  = cm_arr[:, r:].sum()
            A  = cm_arr[r:, :].sum()

        prec = TP / P if P > 0 else 0.0
        rec  = TP / A if A > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics['per_class'][lbl] = {
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
            'support':   int(A)
        }

    if run_rank_correlation:
        true_ranks = []
        pred_means = []
        for i in range(K):
            sup = cm_arr[i, :].sum()
            if sup == 0:
                continue
            mean_pred = (cm_arr[i, :] * np.arange(K)).sum() / sup
            true_ranks.append(i)
            pred_means.append(mean_pred)
        if len(true_ranks) > 1:
            rho, _ = spearmanr(true_ranks, pred_means)
            tau, _ = kendalltau(true_ranks, pred_means)
        else:
            rho, tau = float('nan'), float('nan')
        metrics['spearman_rho'] = float(rho)
        metrics['kendall_tau']  = float(tau)

    return metrics


# ======================================================================
# 5.  database routine
# ======================================================================
def DatabaseTesting(args, db):
    log = logging.getLogger(db)
    log.setLevel(logging.INFO)
    p = Path(args.databases_loc) / db
    log.info("=== %s ===", db)

    # build embeddings
    ani = process_graph(remove_version(pd.read_csv(p / "self_ANI.tsv", sep="\t")))
    hyp = process_graph(remove_version(pd.read_csv(p / "hypergeom_edges.csv")))
    emb = build_embeddings(ani, hyp, args.threads)

    # split nodes
    all_nodes = list(emb.keys())
    random.shuffle(all_nodes)
    c1, c2 = int(0.8 * len(all_nodes)), int(0.9 * len(all_nodes))
    splits = {
        "train": all_nodes[:c1],
        "val"  : all_nodes[c1:c2],
        "test" : all_nodes[c2:],
    }

    # load & split relationships
    meta_df = pd.read_csv(p / f"{db}.csv")
    rel_df  = build_rel_bounds(meta_df, score_config[db])
    edges = {
        k: rel_df[
            rel_df["source"].isin(splits[k]) &
            rel_df["target"].isin(splits[k])
        ].reset_index(drop=True)
        for k in splits
    }

    per_cls = {
        "train": NUM_PER_CLASS,
        "val"  : NUM_PER_CLASS // 4,
        "test" : NUM_PER_CLASS // 4,
    }
    rng = np.random.default_rng(RNG_SEED)

    tris = {
        k: sample_intra_split_triangles(
               splits[k], edges[k], per_cls[k], K_CLASSES, rng
           )
        for k in splits
    }
    for k in tris:
        log.info("%s triangles = %d", k, len(tris[k]))

    ds = {k: TriDS(v, emb) for k, v in tris.items()}
    ld = {
        k: DataLoader(ds[k], batch_size=BATCH_SIZE, shuffle=(k == "train"))
        for k in ds
    }

    model = OrdTri(COMB_DIM, K_CLASSES).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    log.info("Training …")
    for ep in range(1, EPOCHS + 1):
        trL, trH, trA, _ = run_epoch(model, ld["train"], opt)
        vaL, vaH, vaA, _ = run_epoch(model, ld["val"])
        log.info(
            "Ep%02d  train L=%.3f hit=%.3f acc=%.3f   val L=%.3f hit=%.3f acc=%.3f",
            ep, trL, trH, trA, vaL, vaH, vaA
        )

    results = {
        k: run_epoch(model, ld[k], collect_cm=True)
        for k in ("train", "val", "test")
    }

    levels = list(score_config[db].keys())
    for k, (_, _, _, CM) in results.items():
        print(f"\n{k.upper()} confusion matrix (true rows / pred cols):")
        if CM is None:
            print("  — no point‐label edges —")
        else:
            df = pd.DataFrame(CM, index=levels, columns=levels)
            print(df.to_string())

    print(
        f"\n{db}  hit="
        f"{results['train'][1]:.3f}/"
        f"{results['val'][1]:.3f}/"
        f"{results['test'][1]:.3f}  "
        f"acc="
        f"{results['train'][2]:.3f}/"
        f"{results['val'][2]:.3f}/"
        f"{results['test'][2]:.3f}"
    )

    if EVALUATION_METRICS_ENABLED:
        for split in ("train", "val", "test"):
            CM = results[split][3]
            if CM is None:
                continue
            extra = compute_eval_cm_metrics(CM, levels)
            print(f"\n{split.upper()} extra hierarchical metrics:")
            for lvl, m in extra['per_class'].items():
                print(f"  {lvl}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, support={m['support']}")
            print(f"  Spearman rho: {extra['spearman_rho']:.3f}, Kendall tau: {extra['kendall_tau']:.3f}")


# ======================================================================
# 6.  TestHandler
# ======================================================================
def TestHandler(args):
    dbs = database_info()["Class"] if args.all else [args.database]
    for db in dbs:
        if db == "VOGDB":
            continue
        DatabaseTesting(args, db)
