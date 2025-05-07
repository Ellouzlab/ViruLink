#!/usr/bin/env python3
# Ordinal-relationship prediction with interval-censored losses
# Triangle sampler + Node2Vec embeddings + raw ANI/HYP edge features
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

# ViruLink utilities
from ViruLink.setup.score_profile  import score_config
from ViruLink.setup.run_parameters import parameters
from ViruLink.setup.databases      import database_info
from ViruLink.relations.relationship_edges import build_relationship_edges
from ViruLink.train.OrdTri import OrdTri
from ViruLink.train.n2v import n2v
from ViruLink.train.losses import CUM_TRE_LOSS, _adjacent_probs
from ViruLink.sampler.triangle_sampler import sample_triangles
from ViruLink.utils import logging_header



# ─────────────────────Hyperparameters───────────────────────────────
RNG_SEED      = 42
EPOCHS        = 20
BATCH_SIZE    = 512
LR            = 1e-3

NUM_PER_CLASS = 4000          # triangles per (rank × upper|lower)
LAMBDA_INT    = 1.0
LAMBDA_TRI    = 0.0 # ideal case 0.0. Improvement @ 0.1 but only with higher edge pred count (4)

EMBED_DIM     = parameters["embedding_dim"]
COMB_DIM      = EMBED_DIM * 2        # after fusing ANI+HYP embeddings
EVALUATION_METRICS_ENABLED = True

# rank helpers
LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config["Caudoviricetes"])}
K_CLASSES  = max(LEVEL2RANK.values()) + 1
NR_CODE    = LEVEL2RANK["NR"]               # least specific rank
RANK2LEVEL = {r: lvl for lvl, r in LEVEL2RANK.items()}
EDGE_ORDER        = ("r1r2", "qr2", "qr1")   # prediction sequence
EDGE_PRED_COUNT   = 4

# device & RNG 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RNG_SEED)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ──────────────────────────BEGIN!!───────────────────────────────


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


# taxonomy helper
def build_rel_bounds(meta_df: pd.DataFrame,
                     rel_scores: Dict[str, int]) -> pd.DataFrame:
    df = build_relationship_edges(meta_df, rel_scores)
    return (df.rename(columns={"rank_low": "lower", "rank_up": "upper"})
              .astype({"lower": "uint8", "upper": "uint8"})
              [["source", "target", "lower", "upper"]])

# triangle sampler
def sample_intra_split_triangles(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_class: int,
    k_classes: int,
    threads: int,
    seed: int = RNG_SEED
) -> List[Tuple]:
    """
    Fast C++ implementation (pybind11/OpenMP).

    Parameters
    ----------
    nodes          : list of node IDs that belong to *this* split
    rel_df         : dataframe with columns [source, target, lower, upper]
    num_per_class  : triangles per (rank × upper|lower)
    k_classes      : number of rank levels
    threads        : #threads for OpenMP (0 => default)
    seed           : RNG seed passed to C++ sampler
    """
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


# dataset + model
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


# losses + metrics
def interval_hit_rate(logits, bounds):
    """
    Percentage of samples whose **predicted rank** lies inside
    the ground-truth interval [lower, upper] (inclusive).
    """
    pred = torch.argmax(_adjacent_probs(logits), dim=1)
    lo   = bounds[:, 0]
    up   = bounds[:, 1]
    hits = (pred >= lo) & (pred <= up)                    # bool mask
    return hits.float().mean().item()

def run_epoch(model, loader, opt=None, collect_cm=False):
    """
    One training / validation epoch.

    Workflow
    --------
    1. Start with uniform rank-probability vectors for qr1, qr2, r1r2.
    2. For EDGE_PRED_COUNT cycles:
         - predict edges in EDGE_ORDER
         - update that edge’s probability vector with its predicted distribution.
    3. Compute the loss from the final logits of all three edges.
    4. Metrics:
         – hit-rate (all samples, edge qr1)
         – accuracy on point labelled qr1 edges only (upper == lower)
    """
    totL = totH = 0.0            # epoch totals for loss, hit
    totA = 0.0                   # aggregated accuracy *only for point edges*
    n    = 0                     # total number of samples
    n_pt = 0                     # total number of point-label samples

    tbuf, pbuf = [], []          # confusion-matrix buffers
    model.train(opt is not None)

    for batch in loader:
        # send everything to device
        for k in batch:
            batch[k] = batch[k].to(device)

        B = batch["eq"].size(0)          # mini-batch size

        # 1.  initialise uniform priors for the three edges
        uniform = torch.full((B, K_CLASSES), 1.0 / K_CLASSES, device=device)
        cur_p = {"qr1": uniform.clone(),
                 "qr2": uniform.clone(),
                 "r1r2": uniform.clone()}

        last_logits = {}

        # 2.  sequential prediction cycles 
        for _ in range(EDGE_PRED_COUNT):
            for edge_key in EDGE_ORDER:
                # select node-pair embeddings for this edge
                if edge_key == "qr1":
                    xa, xb = batch["eq"], batch["ea"]
                elif edge_key == "qr2":
                    xa, xb = batch["eq"], batch["eh"]
                else:                       # "r1r2"
                    xa, xb = batch["ea"], batch["eh"]

                probs_vec = torch.cat(
                    [cur_p["qr1"], cur_p["qr2"], cur_p["r1r2"]], dim=1
                )                                       # [B, 3·K]

                lg = model(xa, xb, batch["edge"], probs_vec)
                last_logits[edge_key] = lg

                # update probabilities for this edge
                cur_p[edge_key] = _adjacent_probs(lg)

        # 3.  compute loss from final logits
        la = last_logits["qr1"]
        lh = last_logits["qr2"]
        lr = last_logits["r1r2"]

        loss = CUM_TRE_LOSS(la, lh, lr, batch,
                            LAMBDA_INT=LAMBDA_INT,
                            LAMBDA_TRI=LAMBDA_TRI)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 4.  metrics 
        hit = interval_hit_rate(la, batch["lqa"])

        # point label mask (upper == lower for qr1 edge)
        mask = batch["lqa"][:, 0] == batch["lqa"][:, 1]
        p_cnt = mask.sum().item()

        if p_cnt:
            preds = torch.argmax(_adjacent_probs(la[mask]), 1)
            acc   = (preds == batch["lqa"][mask, 0]).float().mean().item()

            totA += acc * p_cnt           # weighted by point sample count
            n_pt += p_cnt

            if collect_cm:
                tbuf.extend(batch["lqa"][mask, 0].cpu().tolist())
                pbuf.extend(preds.cpu().tolist())

        # aggregate totals
        totL += loss.item() * B
        totH += hit * B
        n    += B

    cm = (confusion_matrix(tbuf, pbuf,
                           labels=list(range(K_CLASSES)))
          if tbuf else None)

    meanA = totA / n_pt if n_pt else float("nan")

    return totL / n, totH / n, meanA, cm


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



# database routine + Cli
def DatabaseTesting(args, db):
    p=Path(args.databases_loc)/db

    logging_header("Loading %s database", db)
    ani = process_graph(remove_version(pd.read_csv(p/"self_ANI.tsv", sep="\t")))
    hyp = process_graph(remove_version(pd.read_csv(p/"hypergeom_edges.csv")))
    logging.info("ANI Graph: %d edges", len(ani))
    logging.info("HYP Graph: %d edges", len(hyp))
    

    logging_header("Performing node2vec")
    ani_emb = n2v(ani,args.threads, parameters)
    hyp_emb = n2v(hyp,args.threads, parameters)
    emb = fuse_emb(ani_emb, hyp_emb)
    logging.info("Node2vec embeddings generated for %d nodes", len(emb))

    logging_header("Splitting nodes int train/val/test")
    all_nodes=list(emb); random.shuffle(all_nodes)
    c1,c2=int(.8*len(all_nodes)), int(.9*len(all_nodes))
    splits={"train":all_nodes[:c1],"val":all_nodes[c1:c2],"test":all_nodes[c2:]}
    logging.info("train/val/test sizes: %d/%d/%d", len(splits["train"]),
                 len(splits["val"]), len(splits["test"]))

    logging_header("Loading %s relationships", db)
    meta=pd.read_csv(p/f"{db}.csv")
    rel = build_rel_bounds(meta, score_config[db])
    edges={k:rel[rel["source"].isin(splits[k]) & rel["target"].isin(splits[k])]
           .reset_index(drop=True) for k in splits}
    logging.info("Relationship edges: train - %d | val - %d | test - %d", len(edges["train"]),
                 len(edges["val"]), len(edges["test"]))

    logging_header("Sampling Triangles")
    logging.info("This may take a while …")
    tris={k:sample_intra_split_triangles(splits[k],edges[k],
         NUM_PER_CLASS if k=="train" else NUM_PER_CLASS//4,
         K_CLASSES, args.threads,RNG_SEED) for k in splits}
    for k in tris: logging.info("%s triangles = %d",k,len(tris[k]))

    logging.info("Processing triangles")
    ds={k:TriDS(tris[k],emb,ani,hyp) for k in tris}
    ld={k:DataLoader(ds[k],BATCH_SIZE,shuffle=(k=="train")) for k in ds}

    model=OrdTri(COMB_DIM,K_CLASSES).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    
    logging_header("Training Edge Predictor")
    for ep in range(1,EPOCHS+1):
        trL,trH,trA,_=run_epoch(model,ld["train"],opt)
        vaL,vaH,vaA,_=run_epoch(model,ld["val"])
        logging.info("Ep%02d  train L=%.3f hit=%.3f acc=%.3f   val L=%.3f hit=%.3f acc=%.3f",
                 ep,trL,trH,trA,vaL,vaH,vaA)

    results={k:run_epoch(model,ld[k],collect_cm=True) for k in ("train","val","test")}
    levels=list(score_config[db])

    for k,(_,_,_,CM) in results.items():
        logging.info("  — no point-label edges —" if CM is None
              else f"{k.upper()} confusion matrix:\n{pd.DataFrame(CM,index=levels,columns=levels).to_string()}")

    logging.info(f"\n{db}  hit={results['train'][1]:.3f}/{results['val'][1]:.3f}/{results['test'][1]:.3f}  "
          f"acc={results['train'][2]:.3f}/{results['val'][2]:.3f}/{results['test'][2]:.3f}")

    if EVALUATION_METRICS_ENABLED:
        for split in ("train","val","test"):
            CM=results[split][3];  print()
            if CM is None: continue
            extra=compute_eval_cm_metrics(CM,levels)
            logging.info(f"{split.upper()} hierarchical metrics:")
            for lvl,m in extra['per_class'].items():
                logging.info(f"  {lvl}: prec={m['precision']:.3f}, rec={m['recall']:.3f}, "
                      f"f1={m['f1']:.3f}, sup={m['support']}")
            logging.info(f"  Spearman rho={extra['spearman_rho']:.3f}, "
                  f"Kendall tau={extra['kendall_tau']:.3f}")
    logging_header("Finished %s database", db)

def TestHandler(args):
    dbs=database_info()["Class"] if args.all else [args.database]
    for db in dbs:
        if db=="VOGDB": continue
        DatabaseTesting(args,db)
