import logging
import os
import shutil
import requests
import tarfile
from tqdm import tqdm
from Bio import SeqIO


def merge_vogs(folder_path, output_file):
    '''
    Consolidate all VOGs from .faa files in a folder into a single FASTA file.
    
    Each record ID is set to the filename (without extension), and descriptions are cleared.
    
    Args:
        folder_path (str): Path to the folder containing .faa files.
        output_file (str): Path to the output consolidated FASTA file.
    
    Returns:
        None
    '''
    logging.info(f"Merging VOGs from folder: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    with open(output_file, "w") as out_handle:
        for filename in os.listdir(folder_path):
            if filename.endswith(".faa"):
                file_path = os.path.join(folder_path, filename)
                file_id = os.path.splitext(filename)[0]
                
                try:
                    record = next(SeqIO.parse(file_path, "fasta"))
                except StopIteration:
                    logging.warning(f"No records found in file: {file_path}")
                    continue  # Skip files with no records
                
                record.id = file_id
                record.description = ""
                
                SeqIO.write(record, out_handle, "fasta")
                logging.debug(f"Added record: {record.id}")
    
    logging.info(f"Consolidated FASTA file created at {output_file}")


def vogdb_download(vogdb_url, output_dir):
    """
    Download and extract the VOG database from the given URL with a progress bar.
    
    Args:
        vogdb_url (str): URL to the VOG database tarball.
        output_dir (str): Directory where the downloaded file will be extracted.
    
    Returns:
        None
    """
    tarball_name = "vog.faa.tar.gz"
    tarball_path = os.path.join(output_dir, tarball_name)
    
    logging.info(f"Starting download of VOG database from {vogdb_url}...")
    
    try:
        response = requests.get(vogdb_url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download VOG database: {e}")
        raise
    
    total_size = int(response.headers.get('content-length', 0))
    
    # Download the file with a progress bar
    try:
        with open(tarball_path, 'wb') as f, tqdm(
            desc="Downloading VOGDB",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    except Exception as e:
        logging.error(f"Error during download: {e}")
        raise
    
    logging.info(f"Downloaded tarball to {tarball_path}")
    
    # Extract the tarball
    try:
        logging.info("Extracting the tarball...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        logging.info(f"Extracted files to {output_dir}")
    except tarfile.TarError as e:
        logging.error(f"Failed to extract tarball: {e}")
        raise
    
    # Remove the tarball after extraction
    try:
        logging.info(f"Removing the tarball: {tarball_path}")
        os.remove(tarball_path)
        logging.info(f"Removed the tarball {tarball_path}")
    except OSError as e:
        logging.warning(f"Could not remove tarball {tarball_path}: {e}")
    
    # Merge VOGs
    merged_vog_path = os.path.join(output_dir, "vogdb_merged.faa")
    faa_folder = os.path.join(output_dir, "faa")
    
    try:
        merge_vogs(faa_folder, merged_vog_path)
    except Exception as e:
        logging.error(f"Failed to merge VOGs: {e}")
        raise
    
    # Delete the faa folder after merging
    try:
        logging.info(f"Deleting the faa folder: {faa_folder}")
        shutil.rmtree(faa_folder)
        logging.info(f"Deleted the faa folder {faa_folder}")
    except OSError as e:
        logging.warning(f"Could not delete faa folder {faa_folder}: {e}")
    
    logging.info(f"Consolidated VOG database created at {merged_vog_path}")
