#!/usr/bin/env python3
"""
concat_by_assembly.py

Given a FASTA file and a CSV file that contains columns
  • Accession – primary sequence identifier (FASTA headers may carry a “.version” suffix)
  • Assembly   – assembly / biosample ID (may be blank/NA)

…this script

1. For every non‑empty Assembly value that appears on ≥2 rows,
   concatenates the corresponding FASTA sequences **in the CSV row order**
   into a single record whose ID is that of the first accession in the group.
2. Writes a new FASTA containing:
      • the concatenated records (one per assembly),
      • unchanged sequences whose Assembly is unique or empty/NA.
3. Writes a new CSV with duplicates removed w.r.t. Assembly,
   keeping only the first occurrence (so the Accession matches the
   sequence ID used in the output FASTA).

Dependencies:  ▸ pandas  ▸ biopython   (pip install pandas biopython)

"""

from pathlib import Path
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ───────────────────────── helper functions ──────────────────────────
def strip_version(acc: str) -> str:
    """Return accession without trailing '.n' version."""
    return acc.split(".", 1)[0]


def load_fasta_dict(fasta_path: Path) -> dict:
    """Map base accession → SeqRecord (first occurrence wins)."""
    records = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        base = strip_version(rec.id)
        # If the same accession re‑appears we ignore later copies
        records.setdefault(base, rec)
    return records


def normalise_assembly(series: pd.Series) -> pd.Series:
    """Turn '', 'NA', 'None', etc. into pandas NA."""
    s = series.replace({"": pd.NA, "NA": pd.NA, "na": pd.NA,
                        "None": pd.NA, "none": pd.NA})
    return s.astype("string")


# ───────────────────────── core processing ───────────────────────────
def concat_by_assembly(fasta_in: Path, csv_in: Path,
                       fasta_out: Path, csv_out: Path) -> None:
    fasta_dict = load_fasta_dict(fasta_in)

    df = pd.read_csv(csv_in)
    if "Assembly" not in df.columns or "Accession" not in df.columns:
        raise ValueError("CSV must have 'Accession' and 'Assembly' columns")

    df["Assembly"] = normalise_assembly(df["Assembly"])

    out_records = []                      # SeqRecords for output FASTA
    keep_rows    = []                     # rows for output CSV (first in each assembly)

    # Process assemblies that have duplicates (size ≥ 2)
    grouped = df.groupby("Assembly", dropna=True, sort=False)
    handled_bases = set()

    for asm, sub in grouped:
        if len(sub) == 1:
            continue                      # unique assembly – skip for now

        # Gather SeqRecords in CSV order
        recs = []
        for acc in sub["Accession"]:
            base = strip_version(acc)
            rec = fasta_dict.get(base)
            if rec is None:
                raise KeyError(f"Accession {acc} (base {base}) not found in FASTA")
            recs.append(rec)
            handled_bases.add(base)

        # Concatenate sequences
        cat_seq = "".join(str(r.seq) for r in recs)
        new_rec = SeqRecord(
            Seq(cat_seq),
            id=recs[0].id,                        # keep full first header (incl. version)
            description=f"concatenated assembly {asm} ({len(recs)} parts)"
        )
        out_records.append(new_rec)

        # Keep only the first CSV row for this assembly
        keep_rows.append(sub.iloc[0])

    # Add sequences whose assembly is unique or blank/NA
    for idx, row in df.iterrows():
        asm = row["Assembly"]
        acc_base = strip_version(row["Accession"])
        if pd.isna(asm) or grouped.size().get(asm, 0) == 1:
            if acc_base not in handled_bases:
                rec = fasta_dict.get(acc_base)
                if rec is None:
                    raise KeyError(f"Accession {row['Accession']} not found in FASTA")
                out_records.append(rec)
                keep_rows.append(row)

    # Save outputs
    SeqIO.write(out_records, fasta_out, "fasta")
    pd.DataFrame(keep_rows).to_csv(csv_out, index=False)

    print(f"Wrote {len(out_records)} sequences → {fasta_out}")
    print(f"Wrote {len(keep_rows)} CSV rows   → {csv_out}")


# ───────────────────────────────── cli ────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Concatenate FASTA records that share the same Assembly "
                    "and deduplicate the corresponding CSV rows."
    )
    ap.add_argument("--fasta", required=True, type=Path,
                    help="Input FASTA file")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Input CSV file (must contain Accession & Assembly columns)")
    ap.add_argument("--out-fasta", default="concatenated.fasta", type=Path,
                    help="Output FASTA (default: concatenated.fasta)")
    ap.add_argument("--out-csv", default="dedup.csv", type=Path,
                    help="Output CSV (default: dedup.csv)")
    args = ap.parse_args()

    concat_by_assembly(args.fasta, args.csv, args.out_fasta, args.out_csv)


if __name__ == "__main__":
    main()
