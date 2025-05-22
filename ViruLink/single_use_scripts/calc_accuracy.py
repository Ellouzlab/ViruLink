#!/usr/bin/env python3
"""
calc_accuracy.py

Compute macro-averaged precision / recall / F1 for Family, Subfamily,
and Genus predictions, plus a per-class breakdown (top 10 by support).

Rules
-----
* Accepts CSV or Excel for answers/predictions.
* Removes accession version suffixes.
* Normalises blank / Unknown / Unclassified → "Unclassified".
* Rows where the true label is "Unclassified" but the prediction is not
  are ignored (they don’t count as mistakes).
* **Unclassified is not treated as a class**: any row where the true or
  predicted label is Unclassified is excluded from precision/recall/F1.

Usage
-----
python calc_accuracy.py answers.csv predictions.csv
"""
from pathlib import Path
import argparse

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ────────────────────────── config ──────────────────────────
TAXA    = ["Family", "Subfamily", "Genus"]
UNCLASS = "Unclassified"
TOP_N   = 1000   # how many labels to show in per-class tables

# ───────────────────────── helpers ──────────────────────────
def strip_version(s: pd.Series) -> pd.Series:
    """Remove trailing '.<version>' from accession strings."""
    return s.astype(str).str.split(".", n=1).str[0]

def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame (all columns as strings)."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)

def normalise_columns(df: pd.DataFrame) -> None:
    """Trim whitespace from column names (in-place)."""
    df.columns = df.columns.str.strip()

def normalise_labels(df: pd.DataFrame) -> None:
    """
    Convert NaN / '' / 'Unknown' / 'Unclassified' to a single token,
    and strip whitespace. Operates in-place.
    """
    repl = {"": UNCLASS, "Unknown": UNCLASS, "unclassified": UNCLASS, "Unclassified": UNCLASS}
    for col in TAXA:
        if col not in df:
            raise KeyError(f"Missing expected column {col!r}")
        df[col] = df[col].fillna("").astype(str).str.strip().replace(repl)

# ────────────────────── metrics logic ──────────────────────
def calc_metrics(merged: pd.DataFrame):
    """
    Returns:
      - macro_df: DataFrame of macro-averaged P/R/F1 (indexed by TAXA)
      - per_class_tbl: dict rank → per-label DataFrame
    """
    macro_tbl = {}
    per_class_tbl = {}

    for col in TAXA:
        tc = f"{col}_ans"
        pc = f"{col}_pred"

        # ignore rows where true==UNCLASS but pred!=UNCLASS
        mask_ignore = (merged[tc] == UNCLASS) & (merged[pc] != UNCLASS)
        df_all = merged.loc[~mask_ignore, [tc, pc]].copy()

        # exclude any row where true or pred is UNCLASS
        df = df_all[(df_all[tc] != UNCLASS) & (df_all[pc] != UNCLASS)]
        y_true = df[tc]
        y_pred = df[pc]

        # define consistent label order
        labels = sorted(set(y_true) | set(y_pred))

        # per-label metrics (UNCLASS no longer appears)
        p_c, r_c, f1_c, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        per_df = pd.DataFrame({
            "label":     labels,
            "precision": p_c,
            "recall":    r_c,
            "f1":        f1_c,
            "support":   sup
        }).sort_values("support", ascending=False)
        per_class_tbl[col] = per_df

        # macro averages (over non-UNCLASS rows)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_tbl[col] = {"precision": p, "recall": r, "f1": f1}

    macro_df = pd.DataFrame.from_dict(macro_tbl, orient="index")
    return macro_df, per_class_tbl

# ─────────────────────────── main ───────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Taxonomy accuracy metrics")
    ap.add_argument("answers_file", type=Path, help="ground-truth CSV/XLSX")
    ap.add_argument("preds_file",   type=Path, help="predictions CSV/XLSX")
    args = ap.parse_args()

    # load & clean
    answers     = read_table(args.answers_file)
    predictions = read_table(args.preds_file)
    normalise_columns(answers)
    normalise_columns(predictions)

    answers   ["Accession"] = strip_version(answers   ["Accession"])
    predictions["Accession"] = strip_version(predictions["Accession"])
    normalise_labels(answers)
    normalise_labels(predictions)

    # merge on Accession (inner → only predictions kept)
    merged = predictions.merge(
        answers,
        on="Accession",
        how="inner",
        suffixes=("_pred", "_ans")
    )

    # compute metrics
    macro_df, per_class = calc_metrics(merged)

    # display results
    print("=== Macro-averaged precision / recall / F1 ===")
    print(macro_df.to_string(float_format=lambda x: f"{x:.3f}"))
    for col in TAXA:
        print(f"\n--- {col} per-label metrics (top {TOP_N}) ---")
        print(
            per_class[col]
            .head(TOP_N)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

if __name__ == "__main__":
    main()
