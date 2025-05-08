import os
import logging
import pandas as pd
from Bio import SeqIO
import random
from tqdm import tqdm

def get_conversion_dicts(ictv_df):
    fam_to_class = {}
    gen_to_class = {}
    spec_to_class = {}
    
    for index, row in ictv_df.iterrows():
        fam_to_class[row["Family"]] = row["Class"]
        gen_to_class[row["Genus"]] = row["Class"]
        spec_to_class[row["Species"]] = row["Class"]
    
    return fam_to_class, gen_to_class, spec_to_class

def make_class_db(arguments):
    filtered_fasta_path = f"{arguments.output}/filtered_sequences.fasta"
    filtered_acc2assem_path = f"{arguments.output}/filtered_acc2assem.csv"
    merged_df_path = f"{arguments.output}/merged_df.csv"
    stats_path = f"{arguments.output}/class_statistics.csv"

    # Step 0: If filtered_sequences.fasta exists but merged_df.csv is missing, recalculate merged_df
    if os.path.exists(filtered_fasta_path) and not os.path.exists(merged_df_path):
        logging.info(f"Recalculating merged_df from {filtered_fasta_path}...")

        # Load filtered_acc2assem and NCBI data
        if os.path.exists(filtered_acc2assem_path):
            filt_acc2assem = pd.read_csv(filtered_acc2assem_path)
        else:
            logging.error(f"Missing {filtered_acc2assem_path}. Cannot rebuild merged_df.")
            return

        ncbi_df = pd.read_csv(arguments.ncbi_csv)

        # Recreate ICTV-based class mappings
        ictv_df = pd.read_csv(arguments.ictv_csv)
        fam_to_class, gen_to_class, spec_to_class = get_conversion_dicts(ictv_df)
        ncbi_df["Class"] = ncbi_df["Family"].map(fam_to_class)
        ncbi_df["Class"] = ncbi_df["Class"].fillna(ncbi_df["Genus"].map(gen_to_class))
        ncbi_df["Class"] = ncbi_df["Class"].fillna(ncbi_df["Species"].map(spec_to_class))
        ncbi_df["Class"] = ncbi_df["Class"].fillna("Unknown")

        # Merge NCBI data with filtered_acc2assem to create merged_df
        merged_df = ncbi_df.merge(filt_acc2assem, on="Assembly", how="left")

        # Read assemblies retained in filtered_sequences.fasta
        logging.info(f"Reading assemblies from {filtered_fasta_path}")
        retained_assemblies = set()
        with open(filtered_fasta_path, "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                retained_assemblies.add(record.id)

        # Filter merged_df to only include retained assemblies
        filtered_df = merged_df[merged_df["Assembly"].isin(retained_assemblies)]

        # Save the recalculated merged_df
        filtered_df.to_csv(merged_df_path, index=False)
        logging.info(f"Recalculated merged_df saved to {merged_df_path}")
    elif os.path.exists(merged_df_path):
        logging.info(f"merged_df already exists at {merged_df_path}. Skipping recalculation.")
    else:
        logging.error("Neither filtered_sequences.fasta nor merged_df.csv is available. Cannot proceed.")
        return

    # Step 1: Reload filtered_df
    filtered_df = pd.read_csv(merged_df_path)

    # Step 2: Ensure filtered_df contains one row per assembly
    filtered_df = filtered_df.drop_duplicates(subset="Assembly", keep="first")

    # Step 3: Save class statistics
    class_stats = filtered_df["Class"].value_counts().reset_index()
    class_stats.columns = ["Class", "Assembly_Count"]
    class_stats.to_csv(stats_path, index=False)

    logging.info(f"Class statistics saved to {stats_path}")

    # Step 4: Optionally rebuild filtered_sequences.fasta
    if not os.path.exists(filtered_fasta_path) or os.path.getsize(filtered_fasta_path) == 0:
        logging.info("Rebuilding filtered_sequences.fasta...")
        assembly_to_records = {}
        with open(arguments.fasta, "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                accession = record.id.split()[0]
                assembly = filtered_df[filtered_df["Accession"] == accession]["Assembly"].iloc[0]
                if assembly not in assembly_to_records:
                    assembly_to_records[assembly] = []
                assembly_to_records[assembly].append(str(record.seq))

        with open(filtered_fasta_path, "w") as output_fasta:
            for assembly, sequences in assembly_to_records.items():
                merged_sequence = "".join(sequences)
                output_fasta.write(f">{assembly}\n{merged_sequence}\n")
        logging.info(f"Filtered FASTA written to {filtered_fasta_path}")



# Example usage:
class Arguments:
    output = "./output"
    Acc2Assem = "acc2assem.csv"
    ncbi_csv = "ncbi.csv"
    ictv_csv = "ictv.csv"
    fasta = "large_sequences.fasta"
    num_class = 10

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arguments = Arguments()
    os.makedirs(arguments.output, exist_ok=True)
    make_class_db(arguments)
