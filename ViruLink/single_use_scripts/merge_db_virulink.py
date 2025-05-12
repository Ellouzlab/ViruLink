#!/usr/bin/env python3
"""
merge_fasta_csv.py

Merge two FASTA/CSV pairs so that each accession appears at most once
in the output FASTA and once in the output CSV.

• The CSV files must contain a column named “Accession”.
• Version suffixes in FASTA headers (e.g. “ABC123.1”) are stripped
  before deduplication so they match the version‑less accessions.
• If a column is present in only one CSV, it is discarded; the output
  CSV contains only the columns shared by *both* inputs.
"""
from pathlib import Path
import argparse
from typing import Set, List
import pandas as pd
from Bio import SeqIO


# ───────────────────────── helper utilities ──────────────────────────
def strip_version(acc: str) -> str:
    """Return accession without a trailing ‘.n’ version specifier."""
    return acc.split(".", 1)[0]


def read_fasta_unique(path: Path, seen: Set[str]) -> List[SeqIO.SeqRecord]:
    """Read *path* and return records whose base accession is not in *seen*."""
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        base = strip_version(rec.id)
        if base not in seen:
            seen.add(base)
            records.append(rec)
    return records


# ───────────────────────── core workhorse ────────────────────────────
def merge_fasta_csv(fa1: Path, csv1: Path,
                    fa2: Path, csv2: Path,
                    out_fa: Path, out_csv: Path) -> None:
    seen_acc: Set[str] = set()

    # FASTA — deduplicate while reading
    records  = read_fasta_unique(fa1, seen_acc)
    records += read_fasta_unique(fa2, seen_acc)
    SeqIO.write(records, out_fa, "fasta")

    # CSV — load individually so we can find shared columns
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Determine common columns (must include Accession)
    common_cols = df1.columns.intersection(df2.columns)
    if "Accession" not in common_cols:
        raise ValueError("Both CSVs must contain the ‘Accession’ column.")

    # Keep only shared columns
    df1 = df1[common_cols]
    df2 = df2[common_cols]

    # Concatenate and keep rows whose accession is in the merged FASTA
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df["Accession"].isin(seen_acc)]

    # Drop duplicate Accessions, keeping the first occurrence
    df = df.drop_duplicates(subset="Accession", keep="first")

    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(records)} sequences → {out_fa}")
    print(f"Wrote {len(df)} CSV rows     → {out_csv}")


# ───────────────────────────── CLI ───────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge two FASTA/CSV pairs (deduplicate by accession; "
                    "keep only columns present in both CSVs)."
    )
    ap.add_argument("--fasta1", required=True, type=Path, help="First FASTA file")
    ap.add_argument("--csv1",   required=True, type=Path, help="CSV paired with FASTA1")
    ap.add_argument("--fasta2", required=True, type=Path, help="Second FASTA file")
    ap.add_argument("--csv2",   required=True, type=Path, help="CSV paired with FASTA2")
    ap.add_argument("--out-fasta", default="merged.fasta", type=Path,
                    help="Merged FASTA output (default: merged.fasta)")
    ap.add_argument("--out-csv",   default="merged.csv",   type=Path,
                    help="Merged CSV output (default: merged.csv)")
    args = ap.parse_args()

    merge_fasta_csv(
        args.fasta1, args.csv1,
        args.fasta2, args.csv2,
        args.out_fasta, args.out_csv,
    )


if __name__ == "__main__":
    main()
