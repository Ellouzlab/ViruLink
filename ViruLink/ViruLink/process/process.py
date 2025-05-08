from ViruLink.search_utils import DiamondCreateDB, DiamondSearchDB, CreateANISketchFolder, ANIDist
from ViruLink.utils import edge_list_to_presence_absence, compute_hypergeom_weights, create_graph, running_message, get_file_path, logging_header
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

def m8_processor(m8_file, class_data, eval_threshold, bitscore_threshold, edge_list_path, force = False):
    if not os.path.exists(edge_list_path) or os.path.getsize(edge_list_path) == 0  or force:
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
    """
    Run the ViruLink database-preparation pipeline.

    Modes
    -----
    * --all               : process every class found under `databases_loc`
                            (exactly what the old code did).
    * --database <class>  : process a single class (non-VOGDB) **plus** VOGDB,
                            which is always needed for the BLAST step.

    Either --all or --database must be provided.
    """
    paths_to_unprocs   = get_paths_dict(arguments.databases_loc, classes_df)
    database_parameter = database_info()

    # ───────────────────── determine which class(es) to process ──────────────────
    if arguments.all:
        target_classes = [c for c in paths_to_unprocs if c != "VOGDB"]
    elif arguments.database is not None:
        if arguments.database not in paths_to_unprocs:
            raise ValueError(
                f"{arguments.database} not found under {arguments.databases_loc}"
            )
        target_classes = [arguments.database]               # single class
    else:
        raise ValueError("Specify either --all or --database <class>")

    # ───────────────────────────── prepare VOGDB once ────────────────────────────
    logging_header("Prepping VOGDB")
    VOGDB_path = paths_to_unprocs["VOGDB"]
    VOGDB_dmnd = generate_database("VOGDB", VOGDB_path)

    # ───────────────────── VOGDB → target BLAST searches ─────────────────────────
    m8_files = {}
    for class_data in target_classes:
        unproc_path = paths_to_unprocs[class_data]
        fasta_path  = get_file_path(unproc_path, "fasta")

        logging_header(f"Running VOGDB vs {class_data} BLAST")
        m8_files[class_data] = DiamondSearchDB(
            VOGDB_dmnd,
            fasta_path,
            unproc_path,
            arguments.threads,
            force=arguments.force,
        )

    # ─────────────────── edge lists + hypergeometric weights ─────────────────────
    for class_data, m8_file in m8_files.items():
        logging_header(f"Processing {class_data} m8 file")

        edge_list = f"{arguments.databases_loc}/{class_data}/edge_list.txt"
        m8_processor(
            m8_file,
            class_data,
            arguments.eval,
            arguments.bitscore,
            edge_list,
            force=arguments.force,
        )

        # presence/absence matrix
        pa = edge_list_to_presence_absence(edge_list)

        out_hypergeom_edges = (
            f"{arguments.databases_loc}/{class_data}/hypergeom_edges.csv"
        )
        if (
            arguments.force
            or not os.path.exists(out_hypergeom_edges)
            or os.path.getsize(out_hypergeom_edges) == 0
        ):
            logging_header(f"Generating gene-sharing network for {class_data}")
            w_mat = compute_hypergeom_weights(pa, arguments.threads)
            src, dst, w = create_graph(w_mat, threshold=0.0)  # keep w>0
            pd.DataFrame({"source": src, "target": dst, "weight": w}).to_csv(
                out_hypergeom_edges, index=False
            )
        else:
            logging.info(f"Hypergeom edges already exist at {out_hypergeom_edges}")

    # ─────────────────────────────── ANI network ─────────────────────────────────
    for class_data in target_classes:
        database_path       = paths_to_unprocs[class_data]
        ANI_sketch_folder   = f"{database_path}/ANI_sketch"
        fasta_path          = get_file_path(database_path, "fasta")

        params             = database_parameter[
            database_parameter["Class"] == class_data
        ]
        skani_sketch_mode  = params["skani_sketch_mode"].values[0]
        skani_dist_mode    = params["skani_dist_mode"].values[0]

        logging_header(f"Creating ANI network for {class_data}")
        logging.info(
            f"Creating ANI sketches in {ANI_sketch_folder} "
            f"using {skani_sketch_mode} mode."
        )
        sketch_paths_txt = CreateANISketchFolder(
            fasta_path, ANI_sketch_folder, arguments.threads, skani_sketch_mode
        )

        output_path = f"{database_path}/self_ANI.tsv"
        logging.info(f"Calculating ANI distances ({skani_dist_mode} mode).")
        ANIDist(
            sketch_paths_txt,
            sketch_paths_txt,
            output_path,
            arguments.threads,
            skani_dist_mode,
            arguments.force,
            arguments.ANI_FRAC_weights,
        )

                