import os
import gdown
import logging
import sys
import pandas as pd
from glob import glob
from tqdm import tqdm
from Bio.Seq import Seq
from ViraLink.download.vogdb import vogdb_download
from ViraLink.utils import read_fasta, write_fasta

def fix_family(row):
    if pd.isna(row["Family_y"]) and not pd.isna(row["Family_x"]):
        return row["Family_x"]
    
    else:
        return row["Family_y"]

def EnrichCsv(ncbi_df, ictv_csv):
    ictv_df = pd.read_csv(ictv_csv).drop(columns=["Species"])
    
    # Deduplicate ICTV DataFrame
    ictv_df = ictv_df.drop_duplicates(subset=["Genus", "Family"])
    
    ncbi_df["Genus"] = ncbi_df["Genus"].map(lambda x: x if not pd.isnull(x) else "Unknown")
    ncbi_df["Family"] = ncbi_df["Family"].map(lambda x: x if not pd.isnull(x) else "Unknown")
    
    ncbi_mask = (~(ncbi_df["Family"] == "Unknown")) & (ncbi_df["Genus"] == "Unknown")
    ncbi_only_fam = ncbi_df[ncbi_mask]
    ncbi_rest = ncbi_df[~ncbi_mask]
    
    merged_rest = ncbi_rest.merge(ictv_df, how="left", on="Genus")
    
    if "Family_x" in merged_rest.columns:
        merged_rest["Family"] = merged_rest.apply(fix_family, axis=1)
        merged_rest = merged_rest.drop(columns=["Family_x", "Family_y"])
    
    ictv_df = ictv_df.drop(columns=["Genus"])
    merged_only_fam = ncbi_only_fam.merge(ictv_df, how="left", on="Family")
    
    result = pd.concat([merged_rest, merged_only_fam], ignore_index=True)
    
    result.drop_duplicates(subset=["Accession"], keep="first", inplace=True)
    assert len(result) == len(ncbi_df), "Row count mismatch after enrichment"
    return result

    

def ClassDownloadProcessor(class_outpath, folder, ictv_path):
    csv_paths = glob(f"{class_outpath}/*.csv")
    
    initial_csv = [csv for csv in csv_paths if not "complete" in csv]
    
    if len(initial_csv) == 0:
        logging.info(f"Class {class_outpath} missing initial CSV files.")
        logging.info(f"Use --database flag to redownload.")
        sys.exit(1)
    
    fasta_paths = glob(f"{class_outpath}/*.fasta")
    
    if len(fasta_paths) == 0:
        logging.info(f"Class {class_outpath} missing FASTA files.")
        logging.info(f"Use --database flag to redownload.")
        sys.exit(1)
    
    seq_list = read_fasta(fasta_paths[0])
    seq_dict = {seq.id.split('.')[0]: seq for seq in seq_list}
    seq_df = pd.read_csv(initial_csv[0])
    
    if "Assembly" not in seq_df.columns or "Accession" not in seq_df.columns:
        logging.error(f"CSV at {initial_csv[0]} missing required columns 'Assembly' or 'Accession'.")
        sys.exit(1)
    
    representative_records = []
    for assembly, group in seq_df.groupby("Assembly"):
        representative_row = group.iloc[0]
        representative_accession = representative_row["Accession"]
        representative_seq = seq_dict[representative_accession]
        
        concatenated_sequence = ''.join([str(seq_dict[accession].seq) for accession in group["Accession"] if accession in seq_dict])
        representative_seq.seq = Seq(concatenated_sequence)
        
        representative_records.append(representative_seq)
    
    write_fasta(representative_records, fasta_paths[0])
    
    fixed_seq_df = seq_df.drop_duplicates(subset="Assembly", keep="first")
    fixed_seq_df.to_csv(initial_csv[0], index=False)
    logging.info(f"Updated DataFrame saved to {initial_csv[0]}.")
    logging.info(f"Updated FASTA file saved to {fasta_paths[0]}.")
    
    ictv_csv = glob(f"{ictv_path}/*.csv")
    
    new_df = EnrichCsv(fixed_seq_df, ictv_csv[0])
    new_df.to_csv(initial_csv[0], index=False)
    
    

def GoogleDownload(url, output_dir):
    logging.info(f"Downloading {url} to {output_dir}")
    gdown.download_folder(url,output=output_dir,quiet=False)
    logging.info(f"Downloaded {url} to {output_dir}")

def PreparingDownload(row, output_dir, unprocessed, ictv_path):
    to_download = {"unprocessed": row["Unprocessed_url"]}
    if not unprocessed:
        to_download["processed"] = row["Processed_url"]
    
    class_path = os.path.join(output_dir, row["Class"])
    os.makedirs(class_path, exist_ok=True)
    
    for folder, url in to_download.items():
        logging.info(f"Downloading {row['Class']} {folder} data.")
        class_outpath = os.path.join(class_path, folder)
        if not os.path.exists(class_outpath):
            GoogleDownload(url, class_outpath)
        ClassDownloadProcessor(class_outpath, folder, ictv_path)

def DownloadHandler(arguments, classes_df):
    vogdb_url = "https://fileshare.csb.univie.ac.at/vog/vog227/vog.faa.tar.gz"
    ictv_MSL_url = "https://drive.google.com/drive/folders/1okNtAJfBwng1FoRvT5y45PwpyFGv16r3?usp=drive_link"
    
    ictv_path = os.path.join(arguments.output, "ictv")
    if not os.path.exists(ictv_path):
        logging.info(f"Downloading ICTV MSL data to {ictv_path}, since none available.")
        gdown.download_folder(ictv_MSL_url, output=ictv_path, quiet=False)
    
    # Download the VOG database
    if arguments.all or arguments.vogdb:
        vogdb_dir = os.path.join(arguments.output, "VOGDB")
        vogdb_unproc_dir = os.path.join(vogdb_dir, "unprocessed")

        # Download if vogdb command invoked.
        if not os.path.exists(vogdb_unproc_dir) or arguments.vogdb:
            os.makedirs(vogdb_dir, exist_ok=True)
            os.makedirs(vogdb_unproc_dir, exist_ok=True)
            vogdb_download(vogdb_url, vogdb_unproc_dir)
    
    if arguments.all:
        classes_df.apply(PreparingDownload, axis=1, args=(arguments.output, arguments.unprocessed, ictv_path))
    
    if not arguments.database==None:
        for class_name in arguments.database:
            row = classes_df[classes_df["Class"]==arguments.database].iloc[0]
            PreparingDownload(row, arguments.output, arguments.unprocessed)