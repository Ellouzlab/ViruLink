from ViruLink.search_utils import DiamondCreateDB, DiamondSearchDB
from ViruLink.utils import edge_list_to_presence_absence, compute_hypergeom_pvalues, create_graph, running_message
import os, logging, sys
from glob import glob
import pandas as pd
import math

@running_message
def get_paths_dict(databases_loc, classes_df):
    path_to_unprocs = {}
    
    for class_data in classes_df["Class"].to_list():
        class_unproc = f"{databases_loc}/{class_data}" #unprocessed in same folder
        logging.info(f"Checking for {class_data} unprocessed data.")
        
        if os.path.exists(class_unproc):
            logging.info(f"Found {class_data} unprocessed data at {class_unproc}")
            path_to_unprocs[class_data] = class_unproc
        
        else:
            logging.info(f"Could not find {class_data} unprocessed data at {class_unproc}")
            
    VOGDB_unproc = f"{databases_loc}/VOGDB" #unprocessed in same folder
    if os.path.exists(VOGDB_unproc):
        logging.info(f"Found VOGDB unprocessed data at {VOGDB_unproc}")
        path_to_unprocs["VOGDB"] = VOGDB_unproc
    else:
        logging.info(f"Could not find VOGDB unprocessed data at {VOGDB_unproc}")
        logging.info("Please download the VOGDB data.")
        sys.exit(1)
    
    return path_to_unprocs

def get_seqs_path(unproc_path):
    fasta_paths = glob(f"{unproc_path}/*.fasta") + glob(f"{unproc_path}/*.faa")
    
    if len(fasta_paths) == 0:
        logging.error(f"No FASTA files found in {unproc_path}")
        sys.exit(1)
    elif len(fasta_paths) > 1:
        logging.error(f"Multiple FASTA files found in {unproc_path}")
        sys.exit(1)
    else:
        return fasta_paths[0]

def generate_database(class_data, unproc_path):
    db_outpath = f"{unproc_path}/{class_data}"
    DiamondCreateDB(get_seqs_path(unproc_path), db_outpath, force=True)
    return db_outpath

def m8_processor(m8_file, class_data, eval_threshold, bitscore_threshold, edge_list_path):
    if not os.path.exists(edge_list_path):
        columns = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
                "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]

        m8 = pd.read_csv(m8_file, sep="\t", header=None, names=columns)

        # Filter based on thresholds
        filtered_m8 = m8[(m8["evalue"] <= eval_threshold) & (m8["bitscore"] >= bitscore_threshold)]

        # Deduplicate by keeping the best hit
        filtered_m8 = filtered_m8.sort_values(["query", "evalue", "bitscore"], ascending=[True, True, False])
        filtered_m8 = filtered_m8.drop_duplicates(subset=["query", "target"], keep="first")

        # Create edge list
        edge_list = filtered_m8[["query", "target", "qstart", "qend"]]
        edge_list.to_csv(edge_list_path, sep="\t", index=False, header=False)

        logging.info(f"Edge list saved to {edge_list_path}")
        return edge_list_path
    else:
        logging.info(f"Edge list already exists at {edge_list_path}")
        return edge_list_path




def ProcessHandler(arguments, classes_df):
    paths_to_unprocs = get_paths_dict(arguments.databases_loc, classes_df)
    
    if arguments.all:
        
        VOGDB_path = paths_to_unprocs["VOGDB"]
        VOGDB_dmnd = generate_database("VOGDB", VOGDB_path)
        
        m8_files ={}
        for class_data, unproc_path in paths_to_unprocs.items():
            if class_data == "VOGDB":
                continue
            else:
                fasta=glob(f"{unproc_path}/*.fasta")
                if len(fasta) == 0:
                    logging.error(f"No FASTA files found in {unproc_path}")
                    sys.exit(1)
                elif len(fasta) > 1:
                    logging.error(f"Multiple FASTA files found in {unproc_path}")
                    sys.exit(1)
                    
                fasta_path = fasta[0]
                m8_files[class_data] = DiamondSearchDB(VOGDB_dmnd, fasta_path, unproc_path, arguments.threads)
        
        for class_data, m8_file in m8_files.items():    
            meta = pd.read_csv(f"{arguments.databases_loc}/{class_data}/{class_data}.csv") #unprocessed in same
            edge_list = f"{arguments.databases_loc}/{class_data}/edge_list.txt" #processed in same
            
            # Create edge list from the m8 file
            m8_processor(m8_file, class_data, arguments.eval, arguments.bitscore, edge_list)
            
            # Build presence-absence
            pa = edge_list_to_presence_absence(edge_list)
            
            
            out_hypergeom_edges = f"{arguments.databases_loc}/{class_data}/hypergeom_edges.csv"
            if not os.path.exists(out_hypergeom_edges):
                pval = compute_hypergeom_pvalues(pa, arguments.threads)
                sources, destinations, weights = create_graph(pval, threshold = -math.log10(0.05))
                pd.DataFrame({"source": sources, "target": destinations, "weight": weights}).to_csv(f"{arguments.databases_loc}/{class_data}/hypergeom_edges.csv", index=False)