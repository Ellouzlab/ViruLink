import argparse
import pandas as pd
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows based on FASTA IDs (ignoring version numbers).")
    parser.add_argument("--fasta", required=True, help="Input FASTA file with versioned IDs")
    parser.add_argument("--csv", required=True, help="Input CSV file with 'Accession' column (no versions)")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    args = parser.parse_args()

    # Parse FASTA and strip version numbers
    fasta_ids = {record.id.split('.')[0] for record in SeqIO.parse(args.fasta, "fasta")}

    # Read CSV
    df = pd.read_csv(args.csv)

    # Filter based on version-stripped FASTA IDs
    filtered_df = df[df["Accession"].isin(fasta_ids)]

    # Save output
    filtered_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
