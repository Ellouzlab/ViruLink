#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────────────
# Ordinal-relationship prediction with interval-censored losses
# ▸ Evo2 pooled-layer vectors           ▸ Node2Vec on (ANI + HYP) graphs
# ────────────────────────────────────────────────────────────────────────────
# * 50 / 25 / 25 node split (train / val / test) per database class
# * One triangle sample per split, reused for every probing head
# * Evo2: search best of 25 layers (unless --best_layer is given)
# * Node2Vec: walk-embed ANI & HYP individually, concatenate, same head
# * Prints confusion-matrix + hierarchical metrics for both heads
#
# Public entry-point:  TestHandler(args)
# args fields:
#   ─ databases_loc     root of processed ViruLink databases
#   ─ evo2_embeddings   root folder with <Class>/*.npz Evo2 vectors
#   ─ all   (bool)      evaluate every class if True, else args.database
#   ─ database (str)    single class when all==False
#   ─ best_layer (int|None)  choose Evo2 layer  (None ⇒ search)
#   ─ threads (int)     CPU threads for Word2Vec (default os.cpu_count())
# ────────────────────────────────────────────────────────────────────────────
import logging, random, os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# ─── ViruLink helpers ───────────────────────────────────────────────
from ViruLink.setup.score_profile        import score_config
from ViruLink.setup.run_parameters       import parameters          # Node2Vec defaults
from ViruLink.setup.databases            import database_info
from ViruLink.utils                      import (
        prepare_edges_for_cpp, make_all_nodes_list, run_biased_random_walk)
from ViruLink.relations.relationship_edges import build_relationship_edges

# ─── hyper-parameters ───────────────────────────────────────────────
RNG_SEED, EPOCHS, BATCH_SIZE, LR         = 42, 20, 2048, 1e-3
NUM_PER_CLASS, LAMBDA_INT, LAMBDA_TRI    = 4000, 1.0, 0.0

EMBED_DIM                                = parameters["embedding_dim"]  # Node2Vec dim
COMB_DIM                                 = EMBED_DIM * 2                # ANI⊕HYP
EVALUATION_METRICS_ENABLED               = True

# ─── rank helpers (Caudoviricetes schema) ───────────────────────────
LEVEL2RANK = {lvl: r for r, lvl in enumerate(score_config["Caudoviricetes"])}
K_CLASSES  = max(LEVEL2RANK.values()) + 1
NR_CODE    = LEVEL2RANK["NR"]                # most specific rank

# ─── device & RNG ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED); random.seed(RNG_SEED)

# ════════════════════════════════════════════════════════════════════
# 0.  small utilities
# ════════════════════════════════════════════════════════════════════
def strip_version(acc: str) -> str:                   # "NC_012345.1" → "NC_012345"
    return acc.split(".")[0]

# ════════════════════════════════════════════════════════════════════
# 1.  Evo2 helpers
# ════════════════════════════════════════════════════════════════════
def embedding_nodes(vec_dir: Path) -> List[str]:
    """All sequence IDs that have an Evo2 vector file in vec_dir."""
    return [strip_version(p.stem) for p in vec_dir.glob("*.npz")]

def load_evo2_layer_vectors(layer_idx: int,
                            vec_dir: Path) -> Dict[str, np.ndarray]:
    """Return {accession → vector} for a single Evo2 layer."""
    key = f"blocks_{layer_idx}_mlp_l3"
    return {strip_version(p.stem): np.load(p)[key].astype(np.float32)
            for p in vec_dir.glob("*.npz")}

def infer_evo_dim(npz_path: Path) -> int:
    """Infer hidden dimension from any Evo2 .npz file."""
    k = [k for k in np.load(npz_path).files if k.startswith("blocks_0")][0]
    return len(np.load(npz_path)[k])

# ════════════════════════════════════════════════════════════════════
# 2.  Node2Vec helpers
# ════════════════════════════════════════════════════════════════════
def run_walk(df: pd.DataFrame, thr: int):
    r, c, wf, l2id, id2l = prepare_edges_for_cpp(df["source"], df["target"], df["weight"])
    walks = run_biased_random_walk(
        r, c, wf, make_all_nodes_list(l2id),
        parameters["walk_length"], parameters["p"], parameters["q"],
        thr, parameters["walks_per_node"])
    return walks, id2l

def word2vec_emb(walks, id2lbl, thr: int):
    """Gensim Word2Vec → {label → vector}.  Missing → zeros."""
    from gensim.models import Word2Vec
    model = Word2Vec(
        [[str(n) for n in w] for w in walks],
        vector_size=EMBED_DIM,
        window=parameters["window"],
        min_count=0, sg=1, workers=thr, epochs=parameters["epochs"])
    z = np.zeros(EMBED_DIM, np.float32)
    return {id2lbl[idx]: (model.wv[str(idx)] if str(idx) in model.wv else z)
            for idx in id2lbl}

def fuse_emb(ani_emb: Dict[str, np.ndarray],
             hyp_emb: Dict[str, np.ndarray],
             nodes: List[str]) -> Dict[str, np.ndarray]:
    """Concat ANI and HYP embeddings – fill missing with zeros."""
    z = np.zeros(EMBED_DIM, np.float32)
    return {n: np.concatenate([ani_emb.get(n, z), hyp_emb.get(n, z)]).astype(np.float32)
            for n in nodes}

# ════════════════════════════════════════════════════════════════════
# 3.  Graph helpers
# ════════════════════════════════════════════════════════════════════
def remove_version(df):
    for col in ("source", "target"):
        df[col] = df[col].str.split(".").str[0]
    return df

def process_graph(df):
    """→ undirected weighted graph with self-loops at max weight."""
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

# ════════════════════════════════════════════════════════════════════
# 4.  Triangle sampler
# ════════════════════════════════════════════════════════════════════
def sample_intra_split_triangles(
    nodes: List[str],
    rel_df: pd.DataFrame,
    num_per_class: int,
    k_classes: int,
    rng: np.random.Generator
) -> List[Tuple]:
    """
    Same sampler as in the Node2Vec baseline:
      • balanced primary edges across ranks / upper-vs-lower
      • one random ('upper'| 'lower', rank) type for (q,r2)
    Returns list of:
        (q, r1, r2,
         (b1_lo,b1_up), (b2_lo,b2_up), (b3_lo,b3_up))
    """
    # neighbour LUT by (dir,rank)
    up_tbl = rel_df[["source","target","upper"]].rename(columns={"upper":"rank"})
    lo_tbl = rel_df[["source","target","lower"]].rename(columns={"lower":"rank"})
    up_tbl["dir"] = "upper"; lo_tbl["dir"] = "lower"
    tbl = pd.concat([up_tbl, lo_tbl], ignore_index=True)
    tbl = pd.concat([tbl, tbl.rename(columns={"source":"target","target":"source"})],
                    ignore_index=True)
    nbrs = defaultdict(lambda: defaultdict(list))
    for s,t,rnk,d in tbl.itertuples(False):
        nbrs[s][(d,int(rnk))].append(t)

    # primary edge buckets
    key = np.minimum(rel_df["source"], rel_df["target"]) + "|" + \
          np.maximum(rel_df["source"], rel_df["target"])
    lut = dict(zip(key,
                   rel_df[["lower","upper"]].itertuples(False, name=None)))

    upp = {r:[] for r in range(k_classes)}
    low = {r:[] for r in range(k_classes)}
    for s,t,lo,up in rel_df.itertuples(False):
        upp[up].append((s,t)); low[lo].append((s,t))

    prim = []
    for r in range(k_classes):
        prim.extend(random.choices(upp[r], k=num_per_class))
        prim.extend(random.choices(low[r], k=num_per_class))
    df = pd.DataFrame(prim, columns=["q","r1"])
    m = np.random.rand(len(df)) < .5
    df.loc[m,["q","r1"]] = df.loc[m,["r1","q"]].values

    # choose (dir,rank) for r2
    rel_types = [("upper",r) for r in range(k_classes)] + \
                [("lower",r) for r in range(k_classes)]
    t_idx = np.random.randint(0, len(rel_types), len(df))
    dirs  = [rel_types[i][0] for i in t_idx]
    ranks = [rel_types[i][1] for i in t_idx]
    r2, all_nodes = [], np.array(nodes, "U")
    for q,r1,d,rnk in zip(df["q"],df["r1"],dirs,ranks):
        cand = [x for x in nbrs[q].get((d,rnk),[]) if x!=q and x!=r1]
        if cand:
            r2.append(random.choice(cand))
        else:
            x = random.choice(all_nodes)
            while x==q or x==r1: x = random.choice(all_nodes)
            r2.append(x)
    df["r2"] = r2

    # add bounds
    def add(a,b,lo,up):
        k = np.minimum(df[a],df[b]) + "|" + np.maximum(df[a],df[b])
        lo_,up_ = zip(*(lut.get(x,(NR_CODE,NR_CODE)) for x in k))
        df[lo], df[up] = lo_, up_
    add("q","r1","b1_lo","b1_up")
    add("q","r2","b2_lo","b2_up")
    add("r1","r2","b3_lo","b3_up")

    return list(zip(df["q"],df["r1"],df["r2"],
                    zip(df["b1_lo"],df["b1_up"]),
                    zip(df["b2_lo"],df["b2_up"]),
                    zip(df["b3_lo"],df["b3_up"])))

# ════════════════════════════════════════════════════════════════════
# 5.  Dataset
# ════════════════════════════════════════════════════════════════════
class TriDS(Dataset):
    """Triangle dataset returning everything as tensors."""
    def __init__(self,
                 tris: List[Tuple],
                 emb: Dict[str, np.ndarray],
                 ani_df: pd.DataFrame,
                 hyp_df: pd.DataFrame):
        self.t = tris; self.e = emb
        def lut(df):
            d={}
            for s,t,w in df[["source","target","weight"]].itertuples(False):
                d[(s,t)] = w; d[(t,s)] = w
            return d
        self.ani = lut(ani_df); self.hyp = lut(hyp_df)

    def __len__(self): return len(self.t)

    @staticmethod
    def _w(a,b,table): return torch.tensor([table.get((a,b),0.0)],dtype=torch.float32)

    def __getitem__(self, i):
        q,r1,r2,b1,b2,b3 = self.t[i]
        return {
            "eq":   torch.tensor(self.e[q],  dtype=torch.float32),
            "ea":   torch.tensor(self.e[r1], dtype=torch.float32),
            "eh":   torch.tensor(self.e[r2], dtype=torch.float32),
            "edge": torch.cat([
                self._w(q, r1, self.ani), self._w(q, r1, self.hyp),
                self._w(q, r2, self.ani), self._w(q, r2, self.hyp),
                self._w(r1,r2, self.ani), self._w(r1,r2, self.hyp)]),
            "lqa":  torch.tensor(b1, dtype=torch.long),
            "lqh":  torch.tensor(b2, dtype=torch.long),
            "lrr":  torch.tensor(b3, dtype=torch.long),
        }

# ════════════════════════════════════════════════════════════════════
# 6.  Model + losses
# ════════════════════════════════════════════════════════════════════
class OrdTri(nn.Module):
    """Tiny feed-forward ‘head’ that combines two edges with the query node."""
    def __init__(self, dim, k):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(dim*3 + 2 + 6, 128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU())
        self.h1 = nn.Linear(64, k)
        self.h2 = nn.Linear(64, k)

    def _feat(self, q, r, brr, edge):
        return torch.cat([q, r, q-r,
                          brr.float()/(K_CLASSES-1),
                          edge], -1)

    def forward(self, b):
        f1 = self.base(self._feat(b["eq"], b["ea"], b["lrr"], b["edge"]))
        f2 = self.base(self._feat(b["eq"], b["eh"], b["lrr"], b["edge"]))
        return self.h1(f1), self.h2(f2)

# ─── loss pieces & metrics ──────────────────────────────────────────
def interval_ce(logits, bounds):
    p = F.softmax(logits,1); m=torch.zeros_like(p)
    for i,(lo,up) in enumerate(bounds.tolist()): m[i,lo:up+1]=1
    return -(torch.clamp((p*m).sum(1),1e-12).log()).mean()

def exp_rank(l): return (F.softmax(l,1) * torch.arange(K_CLASSES,device=l.device)).sum(1)
def hinge_sq(p,lo,up): return torch.relu(lo-p)**2 + torch.relu(p-up)**2

def tri_dual(r1,r2,lb,ub):
    return (hinge_sq(r1, torch.min(r2,lb.float()), torch.min(r2,ub.float())) +
            hinge_sq(r2, torch.min(r1,lb.float()), torch.min(r1,ub.float()))).mean()

def full_loss(la, lh, b):
    ce  = interval_ce(la,b["lqa"]) + interval_ce(lh,b["lqh"])
    tri = tri_dual(exp_rank(la),exp_rank(lh), b["lrr"][:,0], b["lrr"][:,1])
    return LAMBDA_INT*ce + LAMBDA_TRI*tri

def interval_hit_rate(l,b):
    p=F.softmax(l,1); m=torch.zeros_like(p)
    for i,(lo,up) in enumerate(b.tolist()): m[i,lo:up+1]=1
    return (p*m).sum(1).mean().item()

# ─── epoch runner ───────────────────────────────────────────────────
def run_epoch(model, loader, opt=None, collect_cm=False):
    totL=totH=totA=n=0.; tbuf,pbuf=[],[]
    model.train(opt is not None)
    for batch in loader:
        for k in batch: batch[k]=batch[k].to(device)
        la,lh = model(batch); loss=full_loss(la,lh,batch)
        if opt: opt.zero_grad(); loss.backward(); opt.step()

        hit = interval_hit_rate(la,batch["lqa"])
        mask = batch["lqa"][:,0]==batch["lqa"][:,1]
        acc  = (torch.argmax(la[mask],1)==batch["lqa"][mask,0]).float().mean().item() if mask.any() else float("nan")

        bs = batch["eq"].size(0)
        totL += loss.item()*bs; totH += hit*bs; n += bs
        if not np.isnan(acc): totA += acc*bs
        if collect_cm and mask.any():
            tbuf.extend(batch["lqa"][mask,0].cpu().tolist())
            pbuf.extend(torch.argmax(la[mask],1).cpu().tolist())

    cm = confusion_matrix(tbuf,pbuf,labels=list(range(K_CLASSES))) if collect_cm and tbuf else None
    return totL/n, totH/n, totA/n, cm

# ─── hierarchical metrics helper ────────────────────────────────────
def compute_eval_cm_metrics(cm, labels):
    from scipy.stats import spearmanr, kendalltau
    arr = np.array(cm,int); out={'per_class':{}}
    for r,lbl in enumerate(labels):
        TP = arr[r:,r:].sum() if r!=NR_CODE else arr[r,r]
        P  = arr[:,r:].sum() if r!=NR_CODE else arr[:,r].sum()
        A  = arr[r:,:].sum() if r!=NR_CODE else arr[r,:].sum()
        prec = TP/P if P else 0.; rec = TP/A if A else 0.
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.
        out['per_class'][lbl] = {'precision':prec,'recall':rec,
                                 'f1':f1,'support':int(A)}
    tr,pred=[],[]
    for i in range(len(labels)):
        sup = arr[i,:].sum()
        if not sup: continue
        tr.append(i)
        pred.append((arr[i,:]*np.arange(len(labels))).sum()/sup)
    if len(tr)>1:
        rho,_=spearmanr(tr,pred); tau,_=kendalltau(tr,pred)
    else: rho=tau=float('nan')
    out['spearman_rho']=rho; out['kendall_tau']=tau
    return out

# ════════════════════════════════════════════════════════════════════
# 7.  database routine
# ════════════════════════════════════════════════════════════════════
def DatabaseTesting(args, db):
    log=logging.getLogger(db); log.setLevel(logging.INFO)
    root  = Path(args.databases_loc) / db
    evo_dir = Path(args.evo2_embeddings)
    threads = getattr(args,"threads",os.cpu_count() or 8)
    log.info("=== %s ===", db)

    # ------- nodes / edges ------------------------------------------
    nodes = embedding_nodes(evo_dir)
    if not nodes:
        log.warning("No Evo2 vectors — skipped."); return
    log.info("Embeddings for %d nodes", len(nodes))

    ani = process_graph(remove_version(pd.read_csv(root/"self_ANI.tsv",sep="\t")))
    hyp = process_graph(remove_version(pd.read_csv(root/"hypergeom_edges.csv")))
    ani = ani[ani["source"].isin(nodes)&ani["target"].isin(nodes)].reset_index(drop=True)
    hyp = hyp[hyp["source"].isin(nodes)&hyp["target"].isin(nodes)].reset_index(drop=True)

    # ------- splits --------------------------------------------------
    random.shuffle(nodes)
    c1,c2 = int(.5*len(nodes)), int(.75*len(nodes))
    splits = {"train":nodes[:c1], "val":nodes[c1:c2], "test":nodes[c2:]}
    log.info("train %d  val %d  test %d",
             *(len(splits[k]) for k in ("train","val","test")))

    # ------- relationship bounds ------------------------------------
    meta = pd.read_csv(root/f"{db}.csv")
    acc_col="Accession"; meta[acc_col]=meta[acc_col].map(strip_version)
    meta = meta[meta[acc_col].isin(nodes)].reset_index(drop=True)

    rel = (build_relationship_edges(meta,score_config[db])
           .rename(columns={"rank_low":"lower","rank_up":"upper"})
           .astype({"lower":"uint8","upper":"uint8"})
           [["source","target","lower","upper"]])
    rel = rel[rel["source"].isin(nodes)&rel["target"].isin(nodes)].reset_index(drop=True)
    edges = {k: rel[(rel["source"].isin(splits[k]))&(rel["target"].isin(splits[k]))]
             for k in splits}

    # ------- triangles ----------------------------------------------
    rng=np.random.default_rng(RNG_SEED)
    tris={k:sample_intra_split_triangles(
           splits[k],edges[k],
           NUM_PER_CLASS if k=="train" else NUM_PER_CLASS//4,
           K_CLASSES,rng) for k in splits}
    for k in tris: log.info("%s triangles = %d", k, len(tris[k]))

    # ═════════════════ Evo2 probing ═════════════════════════════════
    EVO_DIM = infer_evo_dim(next(evo_dir.glob("*.npz")))
    target_layer = args.best_layer
    total_layers = 25
    if target_layer is None:
        best_layer,best_hit,best_acc=-1,-1.,-1.
        for layer in range(total_layers):
            emb=load_evo2_layer_vectors(layer,evo_dir)
            if len(emb)<len(nodes): continue
            ds={k:TriDS(tris[k],emb,ani,hyp) for k in splits}
            ld={k:DataLoader(ds[k],BATCH_SIZE,shuffle=(k=="train")) for k in splits}
            model=OrdTri(EVO_DIM,K_CLASSES).to(device); opt=torch.optim.Adam(model.parameters(),lr=LR)
            for _ in range(EPOCHS): run_epoch(model,ld["train"],opt)
            _,hit,acc,_ = run_epoch(model,ld["val"])
            log.info("layer %02d  val-hit=%.3f  val-acc=%.3f", layer, hit, acc)
            if hit>best_hit:
                best_hit,best_layer,best_acc = hit,layer,acc
                best_state = model.state_dict()
        target_layer = best_layer
        log.info("SELECT layer %d  (val hit %.3f  acc %.3f)", best_layer, best_hit, best_acc)
    else:
        emb = load_evo2_layer_vectors(target_layer,evo_dir)
        ds={k:TriDS(tris[k],emb,ani,hyp) for k in splits}
        ld={k:DataLoader(ds[k],BATCH_SIZE,shuffle=(k=="train")) for k in splits}
        model=OrdTri(EVO_DIM,K_CLASSES).to(device); opt=torch.optim.Adam(model.parameters(),lr=LR)
        for _ in range(EPOCHS): run_epoch(model,ld["train"],opt)
        best_state = model.state_dict()

    emb = load_evo2_layer_vectors(target_layer,evo_dir)
    evo_ds={k:TriDS(tris[k],emb,ani,hyp) for k in splits}
    evo_ld={k:DataLoader(evo_ds[k],BATCH_SIZE) for k in splits}
    evo_head=OrdTri(EVO_DIM,K_CLASSES).to(device); evo_head.load_state_dict(best_state)

    # ═════════════════ Node2Vec head ════════════════════════════════
    walks,id2a = run_walk(ani,threads); walks_h,id2h = run_walk(hyp,threads)
    ani_emb = word2vec_emb(walks,id2a,threads)
    hyp_emb = word2vec_emb(walks_h,id2h,threads)
    n2v_emb = fuse_emb(ani_emb,hyp_emb,nodes)

    n2v_ds={k:TriDS(tris[k],n2v_emb,ani,hyp) for k in splits}
    n2v_ld={k:DataLoader(n2v_ds[k],BATCH_SIZE,shuffle=(k=="train")) for k in splits}
    n2v_head=OrdTri(COMB_DIM,K_CLASSES).to(device); opt=torch.optim.Adam(n2v_head.parameters(),lr=LR)
    for _ in range(EPOCHS): run_epoch(n2v_head,n2v_ld["train"],opt)

    # ═════════════════ evaluation printout ══════════════════════════
    def show_confusion(cm,levels):
        print(pd.DataFrame(cm,index=levels,columns=levels).to_string())

    def eval_head(name,hd,ldict):
        print(f"\n{name} results  (layer "
              f"{target_layer if name=='Evo2' else 'Node2Vec'})")
        levels=list(score_config[db])
        for split in ("train","val","test"):
            _,hit,acc,cm = run_epoch(hd,ldict[split],collect_cm=True)
            print(f"{split.upper()}  hit={hit:.3f}  acc={acc:.3f}")
            if cm is not None:
                print("Confusion matrix:"); show_confusion(cm,levels)
                if EVALUATION_METRICS_ENABLED:
                    m = compute_eval_cm_metrics(cm,levels)
                    for lvl,v in m['per_class'].items():
                        print(f" {lvl}: prec={v['precision']:.3f} rec={v['recall']:.3f} "
                              f"f1={v['f1']:.3f} sup={v['support']}")
                    print(f" Spearman ρ={m['spearman_rho']:.3f}  Kendall τ={m['kendall_tau']:.3f}")
        print("-"*60)

    eval_head("Evo2",     evo_head, evo_ld)
    eval_head("Node2Vec", n2v_head, n2v_ld)
    print("="*72+"\n")

# ════════════════════════════════════════════════════════════════════
# 8.  Public entry-point
# ════════════════════════════════════════════════════════════════════
def TestHandler(args):
    """
    args:
        databases_loc   processed-database root
        evo2_embeddings root with Evo2 vectors
        all   (bool)    evaluate all classes if True
        database (str)  single class when all==False
        best_layer (int|None) choose Evo2 layer; None → search
        threads (int)   CPU threads for Word2Vec
    """
    dbs = (database_info()["Class"]
           if getattr(args,"all",False) else [args.database])
    for db in dbs:
        if db == "VOGDB":          # skip VOGDB as in original script
            continue
        DatabaseTesting(args, db)

# ─ optional CLI-less demo ───────────────────────────────────────────
if __name__ == "__main__":
    class Args:
        databases_loc   = "processed_databases"
        evo2_embeddings = "evo2_vectors"
        all             = True
        database        = None
        best_layer      = None      # set e.g. 1 to bypass search
        threads         = 8
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    TestHandler(Args())
