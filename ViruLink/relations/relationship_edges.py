# ViruLink/relations/relationship_edges.py
import numpy as np
import pandas as pd
from .relationship_edges_cpp import compute_pairs   # compiled C++ core


def build_relationship_edges(
    taxa_df: pd.DataFrame,
    rel_scores: dict[str, int],          # e.g. {"NR":0,"Family":1,…}
    unknown_tokens=("Unknown", "", None, np.nan),
    drop_all_nr: bool = False,
) -> pd.DataFrame:
    """
    Return an edge list with numeric relationship codes (uint8).

    Columns
    -------
    source : Accession string
    target : Accession string
    lower  : uint8   (rel_scores[rank] or rel_scores["NR"])
    upper  : uint8
    """

    # ----- 0. rank order (ascending specificity) --------------------
    ranks = sorted((r for r in rel_scores if r != "NR"),
                   key=lambda r: rel_scores[r])

    # ----- 1. normalise DF & encode to int32 codes -----------------
    df = taxa_df[["Accession"] + ranks].copy()
    df.replace(unknown_tokens, np.nan, inplace=True)

    codes = np.column_stack([
        df[r].astype("category").cat.codes.to_numpy(np.int32)
        for r in ranks
    ])

    # ----- 2. call fast C++ kernel ---------------------------------
    src, tgt, lower_i, upper_i = compute_pairs(codes, drop_all_nr)

    # ----- 3. map rank‑indices → tiny uint8 scores -----------------
    #  • rank_index 0…len(ranks)-1   →  rel_scores[ranks[idx]]
    #  • -1 (NR)                    →  rel_scores["NR"]
    rank_scores = np.asarray([rel_scores[r] for r in ranks], dtype=np.uint8)
    nr_code     = np.uint8(rel_scores["NR"])

    lower = np.where(lower_i == -1, nr_code, rank_scores[lower_i]).astype(np.uint8)
    upper = np.where(upper_i == -1, nr_code, rank_scores[upper_i]).astype(np.uint8)

    # ----- 4. assemble DataFrame -----------------------------------
    acc = df["Accession"].to_numpy()          # object dtype but only once
    edges = pd.DataFrame(
        {
            "source": acc[src],
            "target": acc[tgt],
            "lower":  lower,   # uint8
            "upper":  upper,   # uint8
        }
    )

    return edges
