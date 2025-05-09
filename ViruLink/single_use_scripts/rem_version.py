#!/usr/bin/env python3
import os
import argparse
from Bio import SeqIO
from tqdm import tqdm

def read_fasta(fastafile):
    """
    Reads a FASTA file while displaying progress bars for file reading and parsing.
    
    Parameters
    ----------
    fastafile : str
        Path to the FASTA file.

    Returns
    -------
    list
        A list of SeqRecord objects.
    """
    # Get the total size of the file in bytes.
    total_size = os.path.getsize(fastafile)
    
    # First, count the total number of records by scanning through the file.
    with open(fastafile) as f, tqdm(total=total_size, desc="Reading FASTA file", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        total_records = 0
        for line in f:
            pbar.update(len(line))
            if line.startswith(">"):
                total_records += 1

    # Now, parse the FASTA file using SeqIO with a progress bar.
    records = []
    with tqdm(total=total_records, desc="Parsing FASTA file", unit="Record") as pbar:
        for record in SeqIO.parse(fastafile, "fasta"):
            records.append(record)
            pbar.update(1)

    return records

def process_records(records):
    """
    Processes each record by splitting the record ID at the period ('.')
    and using only the first part. Also updates the record name and description.
    
    Parameters
    ----------
    records : list of Bio.SeqRecord.SeqRecord
        List of records read from a FASTA file.
        
    Returns
    -------
    list
        A list of processed SeqRecord objects.
    """
    for record in records:
        # Split the record id at "." and take the first half.
        new_id = record.id.split(".")[0]
        record.id = new_id
        record.name = new_id  # Ensuring name is consistent.
        # Optionally, update description. Here, we replace it with the new id.
        record.description = new_id
    return records

def main():
    parser = argparse.ArgumentParser(
        description="Process a FASTA file by reading records with a progress bar and updating record IDs."
    )
    parser.add_argument(
        "-i", "--infile", required=True,
        help="Input FASTA file."
    )
    parser.add_argument(
        "-o", "--outfile", required=True,
        help="Output FASTA file with updated record IDs."
    )
    
    args = parser.parse_args()
    
    # Read the FASTA file.
    records = read_fasta(args.infile)
    
    # Process the records (update record IDs).
    processed_records = process_records(records)
    
    # Write the processed records to the output FASTA file.
    with open(args.outfile, "w") as out_handle:
        SeqIO.write(processed_records, out_handle, "fasta")
    
    print(f"Processed {len(processed_records)} records and saved to {args.outfile}")

if __name__ == '__main__':
    main()
