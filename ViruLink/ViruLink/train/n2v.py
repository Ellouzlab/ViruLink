import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from ViruLink.utils import (
    prepare_edges_for_cpp,
    make_all_nodes_list,
    run_biased_random_walk,
)


def run_walk(df: pd.DataFrame, thr: int, parameters: dict):
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


def word2vec_emb(walks, id2lbl, thr: int, parameters: dict):
    model = Word2Vec(
        [[str(n) for n in w] for w in walks],
        vector_size=parameters["embedding_dim"],
        window=parameters["window"],
        min_count=0,
        sg=1,
        workers=thr,
        epochs=parameters["epochs"]
    )
    zeros = np.zeros(parameters["embedding_dim"], dtype=np.float32)
    return {lbl: (model.wv[str(idx)] if str(idx) in model.wv else zeros)
            for idx, lbl in id2lbl.items()}
    
def n2v(edge_df: pd.DataFrame, thr: int, parameters: dict):
    edge_w, id2a = run_walk(edge_df, thr, parameters)
    return word2vec_emb(edge_w, id2a, thr, parameters)
