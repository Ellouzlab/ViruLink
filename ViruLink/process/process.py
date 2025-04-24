from ViruLink.search_utils import DiamondCreateDB, DiamondSearchDB, CreateANISketchFolder, ANIDist
from ViruLink.utils import edge_list_to_presence_absence, compute_hypergeom_pvalues, create_graph, running_message, get_file_path
from ViruLink.setup.databases import database_info
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


def generate_database(class_data, unproc_path):
    db_outpath = f"{unproc_path}/{class_data}"
    DiamondCreateDB(get_file_path(unproc_path,"faa"), db_outpath, force=True)
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
    database_parameter = database_info()
    if arguments.all:
        
        VOGDB_path = paths_to_unprocs["VOGDB"]
        VOGDB_dmnd = generate_database("VOGDB", VOGDB_path)
        
        m8_files ={}
        for class_data, unproc_path in paths_to_unprocs.items():
            if class_data == "VOGDB":
                continue
            else:
                fasta_path=get_file_path(unproc_path, "fasta")

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
            else:
                logging.info(f"Hypergeom edges already exist at {out_hypergeom_edges}")
                # Load the existing edges
                edges = pd.read_csv(out_hypergeom_edges)
                
                '''
                # Draw the taxonomic subgraphs
                
                from ViruLink.visualizations import draw_taxonomic_subgraphs
                draw_taxonomic_subgraphs(edges, 1000, seed=42)
                '''
                
        for database, database_path in paths_to_unprocs.items():
            if database == "VOGDB":
                continue
            else:
                ANI_sketch_folder = f"{database_path}/ANI_sketch"
                fasta_path = get_file_path(database_path, "fasta")
                
                # Get the sketch mode from the database_parameter
                parameters = database_parameter[database_parameter["Class"]==database]
                skani_sketch_mode = parameters["skani_sketch_mode"].values[0]
                skani_dist_mode = parameters["skani_dist_mode"].values[0]
                
                logging.info(f"Creating ANI sketches for {database} in {ANI_sketch_folder} using {skani_sketch_mode} mode.")
                sketch_paths_txt = CreateANISketchFolder(fasta_path, ANI_sketch_folder, arguments.threads, skani_sketch_mode)
                
                logging.info(f"Calculating ANI distances for {database} using {skani_dist_mode} mode.")
                output_path = f"{database_path}/self_ANI.tsv"
                ANI_edges = ANIDist(sketch_paths_txt, sketch_paths_txt, output_path, arguments.threads, skani_dist_mode)
                
                '''
                # Draw the taxonomic subgraphs
                
                from ViruLink.visualizations import draw_taxonomic_subgraphs
                draw_taxonomic_subgraphs(edges, 1000, seed=42)
                '''
                