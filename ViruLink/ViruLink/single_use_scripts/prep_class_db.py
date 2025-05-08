import pandas as pd
import os
from Bio import SeqIO
import matplotlib.pyplot as plt

def get_conversion_dicts(ictv_df):
    fam_to_class = {}
    gen_to_class = {}
    spec_to_class = {}
    
    for index, row in ictv_df.iterrows():
        fam_to_class[row["Family"]] = row["Class"]
        gen_to_class[row["Genus"]] = row["Class"]
        spec_to_class[row["Species"]] = row["Class"]
    
    return fam_to_class, gen_to_class, spec_to_class

def prepare_class_db(arguments):
    # Load the ICTV and NCBI data
    ictv_df = pd.read_csv(arguments.ictv_csv)
    ncbi_df = pd.read_csv(arguments.ncbi_csv)
    
    # Generate mapping dictionaries
    fam_to_class, gen_to_class, spec_to_class = get_conversion_dicts(ictv_df)
    
    # Map Class values using the mapping dictionaries
    ncbi_df["Class"] = ncbi_df["Family"].map(fam_to_class)
    ncbi_df["Class"] = ncbi_df["Class"].fillna(ncbi_df["Genus"].map(gen_to_class))
    ncbi_df["Class"] = ncbi_df["Class"].fillna(ncbi_df["Species"].map(spec_to_class))
    
    # Filter rows where both Class and Genus are NaN
    both_nan = ncbi_df[ncbi_df["Class"].isna() & ncbi_df["Genus"].isna()]
    print(f"Number of rows with both Class and Genus NaN: {len(both_nan)}")
    
    ncbi_df = ncbi_df[~(ncbi_df["Class"].isna() & ncbi_df["Genus"].isna())]
    
    # Handle rows with Class as NaN
    class_nan_rows = ncbi_df[ncbi_df["Class"].isna()]
    unique_genus_rows = class_nan_rows.drop_duplicates(subset="Genus").head(50)
    
    # Include a maximum of 50 rows per Class
    ncbi_df = ncbi_df.groupby("Class").head(50)
    ncbi_df = pd.concat([ncbi_df, unique_genus_rows])
    
    # Fill remaining NaN values in Class with "Unknown"
    ncbi_df["Class"] = ncbi_df["Class"].fillna("Unknown")
    
    # Load the Acc2Assem data
    Acc2Assem = pd.read_csv(arguments.Acc2Assem)
    
    # Merge ncbi_df with Acc2Assem to add Accession rows
    merged_df = ncbi_df.merge(Acc2Assem, on="Assembly", how="left")
    
    # Expand rows for each Accession
    expanded_df = merged_df.explode("Accession").reset_index(drop=True)
    print(f"Expanded DataFrame shape: {expanded_df.shape}")
    
    # Save the expanded DataFrame
    os.makedirs(arguments.output, exist_ok=True)
    expanded_df.to_csv(f"{arguments.output}/class_db.csv", index=False)
    
    # Filter FASTA records
    fasta_path = arguments.fasta
    output_fasta_path = f"{arguments.output}/filtered_sequences.fasta"
    filter_fasta_by_ids(fasta_path, output_fasta_path, expanded_df["Accession"])
    
    return expanded_df

def filter_fasta_by_ids(input_fasta, output_fasta, valid_ids):
    """
    Filters a FASTA file to retain only records with IDs in the valid_ids list.
    
    Parameters:
        input_fasta (str): Path to the input FASTA file.
        output_fasta (str): Path to the output filtered FASTA file.
        valid_ids (iterable): A set or list of valid IDs to retain.
    """
    valid_ids_set = set(valid_ids)  # Convert to set for faster lookup
    with open(output_fasta, "w") as out_fasta:
        for record in SeqIO.parse(input_fasta, "fasta"):
            if record.id in valid_ids_set:
                SeqIO.write(record, out_fasta, "fasta")
    print(f"Filtered FASTA saved to {output_fasta}")
    
    