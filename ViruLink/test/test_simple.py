#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Ordinal-relationship prediction (pair-wise, interval-censored)
# Features: Node2Vec embeddings only (no ANI/HYP scalars)
# --------------------------------------------------------------------------
import logging, random, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ---------- ViruLink utilities ------------------------------------------
from ViruLink.setup.score_profile  import score_config
from ViruLink.setup.run_parameters import parameters
from ViruLink.setup.databases      import database_info
from ViruLink.utils import (
    prepare_edges_for_cpp, make_all_nodes_list, run_biased_random_walk
)
from ViruLink.relations.relationship_edges import build_relationship_edges

# ---------- hyper-parameters --------------------------------------------
RNG_SEED      = 42
EPOCHS        = 20
BATCH_SIZE    = 1024
LR            = 1e-3
NUM_PER_CLASS = 4000
EMBED_DIM     = parameters["embedding_dim"]
COMB_DIM      = EMBED_DIM * 2          # ANI-emb ⊕ HYP-emb
EXTRA_METRICS = True                   # hierarchical statistics

# ---------- rank helpers -------------------------------------------------
LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config["Caudoviricetes"])}
K_CLASSES  = max(LEVEL2RANK.values()) + 1
NR_CODE    = LEVEL2RANK["NR"]

# ---------- device & RNG -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RNG_SEED); random.seed(RNG_SEED); np.random.seed(RNG_SEED)

# ======================================================================
# 0.  Edge-table sanitising helpers (unchanged)
# ======================================================================
def clean(df):
    for c in ("source","target"):
        df[c] = df[c].str.split(".").str[0]
    return df

def undirected(df):
    df = df[df["source"]!=df["target"]].copy()
    u = np.minimum(df["source"],df["target"])
    v = np.maximum(df["source"],df["target"])
    df[["u","v"]] = np.column_stack([u,v])
    und = df.groupby(["u","v"],as_index=False)["weight"].max()
    wmax = df["weight"].max()
    nodes = pd.unique(df[["source","target"]].values.ravel())
    self_loops = pd.DataFrame({"u":nodes,"v":nodes,"weight":wmax})
    rev = und.rename(columns={"u":"v","v":"u"})
    return (pd.concat([und,rev,self_loops],ignore_index=True)
              .rename(columns={"u":"source","v":"target"}))

# ======================================================================
# 1.  Node2Vec embedding builders (unchanged)
# ======================================================================
def run_walk(df: pd.DataFrame, thr: int):
    r,c,wf,l2id,id2l = prepare_edges_for_cpp(df["source"], df["target"], df["weight"])
    walks = run_biased_random_walk(
        r, c, wf,
        make_all_nodes_list(l2id),
        parameters["walk_length"],
        parameters["p"], parameters["q"],
        thr, parameters["walks_per_node"]
    )
    return walks, id2l

def word2vec_emb(walks,id2lbl,thr):
    from gensim.models import Word2Vec
    model = Word2Vec(
        [[str(n) for n in w] for w in walks],
        vector_size=EMBED_DIM, window=parameters["window"],
        min_count=0, sg=1, workers=thr, epochs=parameters["epochs"]
    )
    z = np.zeros(EMBED_DIM, np.float32)
    return {lbl: (model.wv[str(i)] if str(i) in model.wv else z)
            for i,lbl in id2lbl.items()}

def fuse(a:Dict[str,np.ndarray], h:Dict[str,np.ndarray]):
    z = np.zeros(EMBED_DIM)
    return {n: np.concatenate([a.get(n,z), h.get(n,z)]) for n in set(a)|set(h)}

def build_embeddings(ani,hyp,thr=8):
    w_a,id2a = run_walk(ani,thr); w_h,id2h = run_walk(hyp,thr)
    return fuse(word2vec_emb(w_a,id2a,thr), word2vec_emb(w_h,id2h,thr))

# ======================================================================
# 2.  bounds helper (unchanged)
# ======================================================================
def bounds_df(meta,scores):
    df = build_relationship_edges(meta,scores)
    return (df.rename(columns={"rank_low":"lower","rank_up":"upper"})
              .astype({"lower":"uint8","upper":"uint8"})
              [["source","target","lower","upper"]])

# ======================================================================
# 3.  pair sampler (unchanged)
# ======================================================================
def sample_pairs(split:List[str], rel:pd.DataFrame, per:int, K:int, rng):
    nodes=set(split)
    rel = rel[rel["source"].isin(nodes)&rel["target"].isin(nodes)].reset_index(drop=True)

    up_gp = rel.groupby("upper").groups
    lo_gp = rel.groupby("lower").groups

    pairs=[]
    for r in range(K):
        for G in (up_gp,lo_gp):
            if r not in G: continue
            block = rel.loc[G[r]][["source","target"]].to_numpy("U")
            m=len(block); take = m if m<=per else per
            sel = rng.choice(m,take,replace=False)
            pairs.extend([tuple(block[i]) for i in sel])

    lut = dict(zip(zip(rel["source"],rel["target"]),
                   rel[["lower","upper"]].itertuples(False,None)))
    lut.update({(b,a):v for (a,b),v in lut.items()})
    return [(a,b,lut.get((a,b),(NR_CODE,NR_CODE))) for a,b in pairs]

# ======================================================================
# 4.  dataset  (embeddings only)
# ======================================================================
class PairDS(Dataset):
    def __init__(self,pairs,emb):
        self.p=np.asarray(pairs,object); self.e=emb
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        q,r,(lo,up)=self.p[i]
        return {"eq":torch.tensor(self.e[q],dtype=torch.float32),
                "er":torch.tensor(self.e[r],dtype=torch.float32),
                "bounds":torch.tensor([lo,up],dtype=torch.long)}

# ======================================================================
# 5.  model + losses  (no edge scalars)
# ======================================================================
class PairNet(nn.Module):
    def __init__(self,dim,k):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim*3,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,k)
        )
    def forward(self,b):
        x=torch.cat([b["eq"], b["er"], b["eq"]-b["er"]], -1)
        return self.net(x)

def interval_ce(logits,bounds):
    p=F.softmax(logits,1); m=torch.zeros_like(p)
    for i,(lo,up) in enumerate(bounds.tolist()):
        m[i,lo:up+1]=1.
    return -(torch.clamp((p*m).sum(1),1e-12).log()).mean()

def interval_hit(logits,bounds):
    p=F.softmax(logits,1); m=torch.zeros_like(p)
    for i,(lo,up) in enumerate(bounds.tolist()):
        m[i,lo:up+1]=1.
    return (p*m).sum(1).mean().item()

# ------------------------------------------------------------------ #
# hierarchical-metric helper (unchanged logic)
# ------------------------------------------------------------------ #
def compute_eval_cm_metrics(cm, labels, run_corr=True):
    from scipy.stats import spearmanr, kendalltau
    arr = np.array(cm,int); K=len(labels)
    metrics={'per_class':{}}
    for r,lbl in enumerate(labels):
        if r==NR_CODE:
            TP=arr[r,r]; P=arr[:,r].sum(); A=arr[r,:].sum()
        else:
            TP=arr[r:,r:].sum(); P=arr[:,r:].sum(); A=arr[r:,:].sum()
        prec=TP/P if P else 0.; rec=TP/A if A else 0.
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        metrics['per_class'][lbl]={'precision':prec,'recall':rec,
                                   'f1':f1,'support':int(A)}
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
# 6.  epoch runner (unchanged)
# ======================================================================
def run_epoch(model,loader,opt=None,collect_cm=False):
    totL=totH=totA=n=n_point=0.
    tbuf,pbuf=[],[]
    model.train(opt is not None)
    for batch in loader:
        for k in batch: batch[k]=batch[k].to(device)
        lg=model(batch); loss=interval_ce(lg,batch["bounds"])
        if opt: opt.zero_grad(); loss.backward(); opt.step()

        hit=interval_hit(lg,batch["bounds"])
        bs=lg.size(0); totL+=loss.item()*bs; totH+=hit*bs; n+=bs

        mask=batch["bounds"][:,0]==batch["bounds"][:,1]
        if mask.any():
            preds=torch.argmax(lg[mask],1)
            gold =batch["bounds"][mask,0]
            acc  =(preds==gold).float().mean().item()
            totA+=acc*mask.sum().item(); n_point+=mask.sum().item()
            if collect_cm:
                tbuf.extend(gold.cpu().tolist())
                pbuf.extend(preds.cpu().tolist())

    cm = (confusion_matrix(tbuf,pbuf,labels=range(K_CLASSES))
          if tbuf else None)
    meanL=totL/n; meanH=totH/n
    acc = totA/n_point if n_point else float('nan')
    return meanL,meanH,acc,cm

# ======================================================================
# 7.  train per database
# ======================================================================
def train_db(args,db):
    log=logging.getLogger(db); log.setLevel(logging.INFO)
    base=Path(args.databases_loc)/db; log.info("=== %s ===",db)

    ani = undirected(clean(pd.read_csv(base/"self_ANI.tsv",sep="\t")))
    hyp = undirected(clean(pd.read_csv(base/"hypergeom_edges.csv")))
    emb = build_embeddings(ani,hyp,args.threads)

    genomes=list(emb); random.shuffle(genomes)
    c1,c2=int(.8*len(genomes)),int(.9*len(genomes))
    split={"train":genomes[:c1],"val":genomes[c1:c2],"test":genomes[c2:]}

    meta=pd.read_csv(base/f"{db}.csv")
    rel=bounds_df(meta,score_config[db])

    rng=np.random.default_rng(RNG_SEED)
    pairs={k:sample_pairs(split[k],rel,
                          NUM_PER_CLASS if k=="train" else NUM_PER_CLASS//4,
                          K_CLASSES,rng)
           for k in split}
    for k in pairs: log.info("%s pairs = %d",k,len(pairs[k]))

    ds={k:PairDS(pairs[k],emb) for k in pairs}
    ld={k:DataLoader(ds[k],BATCH_SIZE,shuffle=(k=="train"),
                     num_workers=4,pin_memory=True) for k in ds}

    net=PairNet(COMB_DIM,K_CLASSES).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=LR)

    log.info("Training …")
    for ep in range(1,EPOCHS+1):
        trL,trH,trA,_ = run_epoch(net,ld["train"],opt)
        vaL,vaH,vaA,_ = run_epoch(net,ld["val"])
        log.info("Ep%02d  L=%.3f/%.3f  hit=%.3f/%.3f  acc=%.3f/%.3f",
                 ep,trL,vaL,trH,vaH,trA,vaA)

    res={k:run_epoch(net,ld[k],collect_cm=True)
         for k in ("train","val","test")}

    levels=list(score_config[db])

    # detailed per-split report
    for split,(L,H,A,CM) in res.items():
        print(f"\n{split.upper()}  loss={L:.3f}  hit={H:.3f}  acc={A:.3f}")
        if CM is None:
            print("  — no point-label edges —")
            continue
        print(pd.DataFrame(CM,index=levels,columns=levels).to_string())

        if EXTRA_METRICS:
            extra = compute_eval_cm_metrics(CM, levels)
            print(f"{split.upper()} hierarchical metrics:")
            for lvl,m in extra['per_class'].items():
                print(f"  {lvl}: prec={m['precision']:.3f}, "
                      f"rec={m['recall']:.3f}, f1={m['f1']:.3f}, "
                      f"sup={m['support']}")
            print(f"  Spearman ρ = {extra['spearman_rho']:.3f}, "
                  f"Kendall τ = {extra['kendall_tau']:.3f}")

    # one-liner summary
    print(f"\n{db}  hit={res['train'][1]:.3f}/{res['val'][1]:.3f}/"
          f"{res['test'][1]:.3f}  "
          f"acc={res['train'][2]:.3f}/{res['val'][2]:.3f}/"
          f"{res['test'][2]:.3f}")

# ======================================================================
# 8.  CLI entry
# ======================================================================
def TestHandler(args):
    dbs = database_info()["Class"] if args.all else [args.database]
    for db in dbs:
        if db=="VOGDB": continue
        train_db(args,db)
