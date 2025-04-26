#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Ordinal-relationship prediction with interval-censored losses
# Triangle sampler + Node2Vec embeddings + raw ANI/HYP edge features
# --------------------------------------------------------------------------
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
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

NUM_PER_CLASS = 4000          # triangles per (rank × upper|lower)
LAMBDA_INT    = 1.0
LAMBDA_TRI    = 0

EMBED_DIM     = parameters["embedding_dim"]
COMB_DIM      = EMBED_DIM * 2        # after fusing ANI+HYP embeddings
EVALUATION_METRICS_ENABLED = True

# ---------- rank helpers -------------------------------------------------
LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config["Caudoviricetes"])}
K_CLASSES  = max(LEVEL2RANK.values()) + 1
NR_CODE    = LEVEL2RANK["NR"]               # most specific rank
RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}

# ---------- device & RNG -------------------------------------------------
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
    return {lbl: (model.wv[str(idx)] if str(idx) in model.wv else zeros)
            for idx, lbl in id2lbl.items()}


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
    return (pd.concat([und, rev, self_loops], ignore_index=True)
              .rename(columns={"u": "source", "v": "target"}))


def fuse_emb(ani_emb: Dict[str, np.ndarray],
             hyp_emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    fused = {}
    for n in set(ani_emb) | set(hyp_emb):
        v1 = ani_emb.get(n, np.zeros(EMBED_DIM))
        v2 = hyp_emb.get(n, np.zeros(EMBED_DIM))
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
    return (df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
              .astype({"lower": "uint8", "upper": "uint8"})
              [["source", "target", "lower", "upper"]])

# ======================================================================
# 3.  triangle sampler
# ======================================================================
def sample_intra_split_triangles(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_class: int,
    k_classes: int,
    rng: np.random.Generator
) -> List[Tuple]:
    """
    1. Pick the primary edge (q,r1) exactly as before:
       • balanced upper / lower buckets
       • guarantees 2·num_per_class triangles per rank.

    2. For each primary q, choose *one* of the 2·k_classes
       relationship types at random:
           ('upper', r)  or  ('lower', r)   for r = 0 … k_classes-1
       Then draw r2 uniformly from the nodes that satisfy that
       relationship with q.  Falls back to the old “any node” rule
       only if q has no neighbour of the chosen type.
    """
    # ------------------------------------------------------------------
    # 0.  Build look-up:  neighbours[q][('upper',rank)] → list[…]
    #                     neighbours[q][('lower',rank)] → list[…]
    # ------------------------------------------------------------------
    upper_tbl = rel_df[["source", "target", "upper"]].rename(columns={"upper": "rank"})
    lower_tbl = rel_df[["source", "target", "lower"]].rename(columns={"lower": "rank"})

    upper_tbl["dir"] = "upper"
    lower_tbl["dir"] = "lower"
    edge_tbl = pd.concat([upper_tbl, lower_tbl], ignore_index=True)

    # make edges undirected
    flipped = edge_tbl.rename(columns={"source": "target", "target": "source"})
    edge_tbl = pd.concat([edge_tbl, flipped], ignore_index=True)

    neighbours: dict[str, dict[Tuple[str, int], list[str]]] = defaultdict(lambda: defaultdict(list))
    for s, t, rnk, d in edge_tbl.itertuples(index=False):
        neighbours[s][(d, int(rnk))].append(t)

    # ------------------------------------------------------------------
    # 1.  Primary pairs (q,r1) – identical to the old code
    # ------------------------------------------------------------------
    rel = rel_df[["source", "target", "lower", "upper"]]
    u = np.minimum(rel["source"], rel["target"])
    v = np.maximum(rel["source"], rel["target"])
    key = u + "|" + v
    lut = dict(zip(key, rel[["lower", "upper"]].itertuples(index=False, name=None)))

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

    # ------------------------------------------------------------------
    # 2.  Choose relationship type for (q,r2) and pick r2
    # ------------------------------------------------------------------
    rel_types = [("upper", r) for r in range(k_classes)] + \
                [("lower", r) for r in range(k_classes)]

    # random type for every row
    type_idx = rng.integers(0, len(rel_types), size=len(dfp))
    chosen_dirs  = np.array([rel_types[i][0] for i in type_idx])
    chosen_ranks = np.array([rel_types[i][1] for i in type_idx])

    r2_list = np.empty(len(dfp), dtype=object)
    all_nodes_arr = np.array(nodes, dtype="U")

    for i, (q, r1) in enumerate(dfp[["q", "r1"]].itertuples(index=False, name=None)):
        key = (chosen_dirs[i], int(chosen_ranks[i]))
        cand = neighbours[q].get(key, [])

        # remove duplicates if they exist
        if cand:
            cand_arr = np.array(cand, dtype="U")
            # exclude q and r1 in vectorised way
            cand_arr = cand_arr[(cand_arr != q) & (cand_arr != r1)]
            if cand_arr.size:
                r2_list[i] = rng.choice(cand_arr)
                continue  # got a valid neighbour

        # Fallback: old behaviour (uniform over nodes ≠ q, r1)
        while True:
            r2 = rng.choice(all_nodes_arr)
            if r2 != q and r2 != r1:
                r2_list[i] = r2
                break

    dfp["r2"] = r2_list

    # ------------------------------------------------------------------
    # 3.  Add interval bounds exactly as before
    # ------------------------------------------------------------------
    def add_bounds(a, b, lo_col, up_col):
        u2 = np.minimum(dfp[a], dfp[b])
        v2 = np.maximum(dfp[a], dfp[b])
        k2 = u2 + "|" + v2
        lo, up = zip(*(lut.get(x, (NR_CODE, NR_CODE)) for x in k2))
        dfp[lo_col] = lo
        dfp[up_col] = up

    add_bounds("q",  "r1", "b1_lo", "b1_up")
    add_bounds("q",  "r2", "b2_lo", "b2_up")
    add_bounds("r1", "r2", "b3_lo", "b3_up")

    # ------------------------------------------------------------------
    # 4.  Return list of triangle tuples (unchanged format)
    # ------------------------------------------------------------------
    return list(zip(
        dfp["q"],  dfp["r1"], dfp["r2"],
        zip(dfp["b1_lo"], dfp["b1_up"]),
        zip(dfp["b2_lo"], dfp["b2_up"]),
        zip(dfp["b3_lo"], dfp["b3_up"])
    ))


# ======================================================================
# 4.  dataset + model
# ======================================================================
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

class OrdTri(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        in_dim = dim * 3 + 2 + 6    # embeddings + interval + raw edges
        self.base = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
        )
        self.h1 = nn.Linear(64, k)
        self.h2 = nn.Linear(64, k)

    def _feat(self, q, r, brr, edge):
        return torch.cat([q, r, q - r,
                          brr.float() / (K_CLASSES - 1),
                          edge], dim=-1)

    def forward(self, b):
        f1 = self.base(self._feat(b["eq"], b["ea"], b["lrr"], b["edge"]))
        f2 = self.base(self._feat(b["eq"], b["eh"], b["lrr"], b["edge"]))
        return self.h1(f1), self.h2(f2)

# ======================================================================
# 5.  losses + metrics
# ======================================================================
def interval_ce(logits, bounds):
    p = F.softmax(logits, dim=1)
    m = torch.zeros_like(p)
    for i, (lo, up) in enumerate(bounds.tolist()):
        m[i, lo:up+1] = 1.0
    return -(torch.clamp((p * m).sum(1), 1e-12).log()).mean()

def exp_rank(logits):
    return (F.softmax(logits, 1) * torch.arange(K_CLASSES, device=logits.device)).sum(1)

def hinge_sq(pred, lo, up):
    return (torch.relu(lo - pred)**2 + torch.relu(pred - up)**2)

def tri_dual(r1, r2, lb, ub):
    l1 = hinge_sq(r1, torch.minimum(r2, lb.float()), torch.minimum(r2, ub.float()))
    l2 = hinge_sq(r2, torch.minimum(r1, lb.float()), torch.minimum(r1, ub.float()))
    return (l1 + l2).mean()

def full_loss(la, lh, b):
    ce  = interval_ce(la, b["lqa"]) + interval_ce(lh, b["lqh"])
    tri = tri_dual(exp_rank(la), exp_rank(lh), b["lrr"][:,0], b["lrr"][:,1])
    return LAMBDA_INT * ce + LAMBDA_TRI * tri

def interval_hit_rate(logits, bounds):
    p = F.softmax(logits, 1)
    m = torch.zeros_like(p)
    for i, (lo, up) in enumerate(bounds.tolist()):
        m[i, lo:up+1] = 1.0
    return (p*m).sum(1).mean().item()

def run_epoch(model, loader, opt=None, collect_cm=False):
    totL = totH = totA = n = 0.
    tbuf, pbuf = [], []
    model.train(opt is not None)

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        la, lh = model(batch)
        loss = full_loss(la, lh, batch)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()

        hit = interval_hit_rate(la, batch["lqa"])
        mask = batch["lqa"][:,0] == batch["lqa"][:,1]
        acc = (torch.argmax(la[mask],1) == batch["lqa"][mask,0]).float().mean().item() \
              if mask.any() else float("nan")

        bs = batch["eq"].size(0)
        totL += loss.item()*bs; totH += hit*bs
        if not np.isnan(acc): totA += acc*bs
        n += bs

        if collect_cm and mask.any():
            tbuf.extend(batch["lqa"][mask,0].cpu().tolist())
            pbuf.extend(torch.argmax(la[mask],1).cpu().tolist())

    cm = confusion_matrix(tbuf, pbuf, labels=list(range(K_CLASSES))) if tbuf else None
    return totL/n, totH/n, totA/n, cm

def compute_eval_cm_metrics(cm, labels, run_corr=True):
    from scipy.stats import spearmanr, kendalltau
    arr = np.array(cm, int); K=len(labels)
    metrics={'per_class':{}}
    for r,lbl in enumerate(labels):
        if r==NR_CODE:
            TP=arr[r,r]; P=arr[:,r].sum(); A=arr[r,:].sum()
        else:
            TP=arr[r:,r:].sum(); P=arr[:,r:].sum(); A=arr[r:,:].sum()
        prec=TP/P if P else 0.; rec=TP/A if A else 0.
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        metrics['per_class'][lbl]={'precision':prec,'recall':rec,'f1':f1,'support':int(A)}
    if run_corr:
        tr,pred=[],[]
        for i in range(K):
            sup=arr[i,:].sum()
            if not sup: continue
            tr.append(i)
            pred.append((arr[i,:]*np.arange(K)).sum()/sup)
        if len(tr)>1:
            rho,_=spearmanr(tr,pred); tau,_=kendalltau(tr,pred)
        else: rho=tau=float('nan')
        metrics['spearman_rho']=rho; metrics['kendall_tau']=tau
    return metrics

# ======================================================================
# 6.  database routine + CLI
# ======================================================================
def DatabaseTesting(args, db):
    log=logging.getLogger(db); log.setLevel(logging.INFO)
    p=Path(args.databases_loc)/db; log.info("=== %s ===",db)

    ani = process_graph(remove_version(pd.read_csv(p/"self_ANI.tsv", sep="\t")))
    hyp = process_graph(remove_version(pd.read_csv(p/"hypergeom_edges.csv")))
    emb = build_embeddings(ani,hyp,args.threads)

    all_nodes=list(emb); random.shuffle(all_nodes)
    c1,c2=int(.8*len(all_nodes)), int(.9*len(all_nodes))
    splits={"train":all_nodes[:c1],"val":all_nodes[c1:c2],"test":all_nodes[c2:]}

    meta=pd.read_csv(p/f"{db}.csv")
    rel = build_rel_bounds(meta, score_config[db])
    edges={k:rel[rel["source"].isin(splits[k]) & rel["target"].isin(splits[k])]
           .reset_index(drop=True) for k in splits}

    rng=np.random.default_rng(RNG_SEED)
    tris={k:sample_intra_split_triangles(splits[k],edges[k],
         NUM_PER_CLASS if k=="train" else NUM_PER_CLASS//4,
         K_CLASSES,rng) for k in splits}
    for k in tris: log.info("%s triangles = %d",k,len(tris[k]))

    ds={k:TriDS(tris[k],emb,ani,hyp) for k in tris}
    ld={k:DataLoader(ds[k],BATCH_SIZE,shuffle=(k=="train")) for k in ds}

    model=OrdTri(COMB_DIM,K_CLASSES).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)

    log.info("Training …")
    for ep in range(1,EPOCHS+1):
        trL,trH,trA,_=run_epoch(model,ld["train"],opt)
        vaL,vaH,vaA,_=run_epoch(model,ld["val"])
        log.info("Ep%02d  train L=%.3f hit=%.3f acc=%.3f   val L=%.3f hit=%.3f acc=%.3f",
                 ep,trL,trH,trA,vaL,vaH,vaA)

    results={k:run_epoch(model,ld[k],collect_cm=True) for k in ("train","val","test")}
    levels=list(score_config[db])

    for k,(_,_,_,CM) in results.items():
        print(f"\n{k.upper()} confusion matrix:")
        print("  — no point-label edges —" if CM is None
              else pd.DataFrame(CM,index=levels,columns=levels).to_string())

    print(f"\n{db}  hit={results['train'][1]:.3f}/{results['val'][1]:.3f}/{results['test'][1]:.3f}  "
          f"acc={results['train'][2]:.3f}/{results['val'][2]:.3f}/{results['test'][2]:.3f}")

    if EVALUATION_METRICS_ENABLED:
        for split in ("train","val","test"):
            CM=results[split][3];  print()
            if CM is None: continue
            extra=compute_eval_cm_metrics(CM,levels)
            print(f"{split.upper()} hierarchical metrics:")
            for lvl,m in extra['per_class'].items():
                print(f"  {lvl}: prec={m['precision']:.3f}, rec={m['recall']:.3f}, "
                      f"f1={m['f1']:.3f}, sup={m['support']}")
            print(f"  Spearman rho={extra['spearman_rho']:.3f}, "
                  f"Kendall tau={extra['kendall_tau']:.3f}")

def TestHandler(args):
    dbs=database_info()["Class"] if args.all else [args.database]
    for db in dbs:
        if db=="VOGDB": continue
        DatabaseTesting(args,db)
