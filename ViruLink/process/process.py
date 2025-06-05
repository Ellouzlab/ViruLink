import os
import logging
import sys
import pandas as pd
import yaml # For loading YAML config

# ViruLink imports
from ViruLink.search_utils import (
    DiamondCreateDB, DiamondSearchDB,
    CreateANISketchFolder, ANIDist,
    CreateDB as MMSeqsCreateDB,
    SearchDBs as MMSeqsSearchDBs,
    m8_file_processor
)
from ViruLink.ani.ani_calc import m8_to_ani # Assumed C++ accelerated version
from ViruLink.utils import (
    edge_list_to_presence_absence, compute_hypergeom_weights, create_graph,
    running_message, get_file_path, logging_header
)
from ViruLink.setup.databases import database_info
from ViruLink.default_yaml import default_yaml_dct

@running_message
def get_paths_dict(databases_loc_arg: str, classes_df_arg: pd.DataFrame) -> dict:
    path_to_unprocs_map = {}
    
    for class_data_name in classes_df_arg["Class"].to_list():
        class_unproc_dir = os.path.join(databases_loc_arg, class_data_name)
        # logging.info(f"Checking for {class_data_name} unprocessed data at {class_unproc_dir}") # Reduced verbosity
        if os.path.exists(class_unproc_dir):
            # logging.info(f"Found {class_data_name} unprocessed data.")
            path_to_unprocs_map[class_data_name] = class_unproc_dir
        else:
            logging.warning(f"Could not find {class_data_name} unprocessed data at {class_unproc_dir}")
            
    vogdb_unproc_dir = os.path.join(databases_loc_arg, "VOGDB")
    # logging.info(f"Checking for VOGDB unprocessed data at {vogdb_unproc_dir}")
    if os.path.exists(vogdb_unproc_dir):
        # logging.info("Found VOGDB unprocessed data.")
        path_to_unprocs_map["VOGDB"] = vogdb_unproc_dir
    else:
        logging.error(f"Could not find VOGDB unprocessed data at {vogdb_unproc_dir}. Please download VOGDB data.")
        sys.exit(1)
    
    return path_to_unprocs_map

def generate_diamond_database(class_name: str, base_unproc_path: str, force_creation: bool) -> str:
    db_out_base_path = os.path.join(base_unproc_path, class_name)
    faa_file = get_file_path(base_unproc_path, "faa", multi=False)
    if not faa_file:
        logging.error(f"Cannot generate Diamond DB for {class_name}, FAA file missing in {base_unproc_path}")
        sys.exit(1) # get_file_path already exits, but for safety.
    DiamondCreateDB(faa_file, db_out_base_path, force=force_creation)
    return db_out_base_path


def ProcessHandler(arguments, classes_df_info: pd.DataFrame):


    unproc_paths_map = get_paths_dict(arguments.databases_loc, classes_df_info)
    db_params_df = database_info()

    if arguments.all:
        classes_to_process = [c for c in unproc_paths_map if c != "VOGDB"]
    elif arguments.database:
        if arguments.database not in unproc_paths_map:
            logging.error(f"DB '{arguments.database}' not found under '{arguments.databases_loc}'.")
            sys.exit(1)
        classes_to_process = [arguments.database]
    else:
        logging.error("Specify --all or --database <class_name>.")
        sys.exit(1)

    logging_header("Preparing VOGDB Diamond Database")
    vogdb_base_path = unproc_paths_map["VOGDB"]

    diamond_m8_output_files = {}
    for target_class_name in classes_to_process:
        
        
        
        # ---------------- Class specific config-----------------------------------#
        logging.info("Loading default YAML configuration for graph making parameters.")
        yaml_config = default_yaml_dct
        
        try:
            graph_making_settings = yaml_config['settings']['normal']['graph_making']
            hypergeom_settings = graph_making_settings['hypergeometric']
            ani_settings = graph_making_settings['ANI']
        except KeyError as e:
            logging.error(f"YAML config missing expected keys: {e}. Using command-line args or defaults.")
            hypergeom_settings = {} 
            ani_settings = {}

        diamond_eval_thresh = float(hypergeom_settings.get('e_value', arguments.eval))
        diamond_bitscore_thresh = float(hypergeom_settings.get('bitscore', arguments.bitscore))
        dimaond_percent_id = hypergeom_settings.get('percent_id', None)
        diamond_db_cov = hypergeom_settings.get('db_cov', None)
        
        if not dimaond_percent_id==None:
            dimaond_percent_id = float(dimaond_percent_id)
        if not diamond_db_cov==None:
            diamond_db_cov = float(diamond_db_cov)
        
        hypergeom_pval_config = float(hypergeom_settings.get('hypergeom_pval', 0.1))
        selected_ani_tool = ani_settings.get('ani_program', 'skani')
        prot_database = hypergeom_settings.get('protein_db', "vogdb")
        use_ani_length_fraction_weight = bool(ani_settings.get('consider_alignment_length', arguments.ANI_FRAC_weights))
        
        
        if prot_database == "vogdb":
            vogdb_diamond_db_path = generate_diamond_database("VOGDB", vogdb_base_path, arguments.force)
        elif prot_database == "custom_prot_db":
            vogdb_diamond_db_path = f"{unproc_paths_map[target_class_name]}/custom_prot_db.dmnd"
            if not os.path.exists(vogdb_diamond_db_path):
                logging.info(f"Could not find custom protein database for {target_class_name} at {vogdb_diamond_db_path}. Utilizing VOGDB instead.")
                vogdb_diamond_db_path = generate_diamond_database(target_class_name, unproc_paths_map[target_class_name], arguments.force)
        # -------------------------------------------------------------------------#
        
        
        

        target_unproc_path = unproc_paths_map[target_class_name]
        target_proteome_fasta = get_file_path(target_unproc_path, "fasta", multi=False)
        if not target_proteome_fasta:
             logging.error(f"No proteome FASTA for {target_class_name} in {target_unproc_path}. Skipping Diamond search.")
             continue

        logging_header(f"Diamond Search: VOGDB vs {target_class_name} Proteome")
        # Corrected call to DiamondSearchDB using positional arguments
        diamond_m8_output_files[target_class_name] = DiamondSearchDB(
            vogdb_diamond_db_path,    # Corresponds to VOGDB_dmnd
            target_proteome_fasta,    # Corresponds to query
            target_unproc_path,       # Corresponds to outdir
            arguments.threads,        # Corresponds to threads
            force=arguments.force,     # Corresponds to force (as keyword for optional)
            percent_id=dimaond_percent_id,  # Optional percent_id
            db_cov=diamond_db_cov,    # Optional db_cov
        )

    for target_class_name, m8_file_path in diamond_m8_output_files.items():
        logging_header(f"Processing m8 Output for {target_class_name}")
        db_class_dir = os.path.join(arguments.databases_loc, target_class_name)
        os.makedirs(db_class_dir, exist_ok=True) # Ensure class directory exists

        vog_edge_list_file = os.path.join(db_class_dir, "edge_list.txt")
        m8_file_processor(
            m8_file_path, diamond_eval_thresh, diamond_bitscore_thresh,
            vog_edge_list_file, force_processing=arguments.force
        )
        print(vog_edge_list_file)
        pa_matrix = edge_list_to_presence_absence(edge_list_path=vog_edge_list_file)
        if pa_matrix.empty:
            logging.warning(f"PA matrix for {target_class_name} empty. Skipping gene-sharing network.")
            continue

        output_hypergeom_edges_file = os.path.join(db_class_dir, "hypergeom_edges.csv")
        if arguments.force or not os.path.exists(output_hypergeom_edges_file) or os.path.getsize(output_hypergeom_edges_file) == 0:
            logging_header(f"Gene-Sharing Network (Hypergeometric-based) for {target_class_name}")
            weight_matrix = compute_hypergeom_weights(
                pa_matrix, arguments.threads, pval_thresh=hypergeom_pval_config, hypergeom=True
            )
            sources, destinations, weights = create_graph(weight_matrix, threshold=0.0)
            pd.DataFrame({"source": sources, "target": destinations, "weight": weights}).to_csv(
                output_hypergeom_edges_file, index=False
            )
        else:
            logging.info(f"Gene-sharing network {output_hypergeom_edges_file} already exists for {target_class_name}.")

    for target_class_name in classes_to_process:
        target_db_base_path = unproc_paths_map[target_class_name]
        genome_fasta_file = get_file_path(target_db_base_path, "fasta", multi=False)
        
        if not genome_fasta_file:
            logging.warning(f"No genome FASTA (fna/fasta) for {target_class_name} in {target_db_base_path}. Skipping ANI.")
            continue
            
        logging_header(f"Creating ANI Network for {target_class_name} using {selected_ani_tool}")

        if selected_ani_tool == 'skani':
            ani_sketch_output_folder = os.path.join(target_db_base_path, "ANI_sketch")
            
            db_specific_params = db_params_df[db_params_df["Class"] == target_class_name]
            skani_sketch_setting = db_specific_params["skani_sketch_mode"].values[0] if not db_specific_params.empty and "skani_sketch_mode" in db_specific_params else ""
            skani_dist_setting = db_specific_params["skani_dist_mode"].values[0] if not db_specific_params.empty and "skani_dist_mode" in db_specific_params else ""
            if not skani_sketch_setting: logging.warning(f"Skani sketch mode for {target_class_name} not in db_info; skani will use defaults.")
            if not skani_dist_setting: logging.warning(f"Skani dist mode for {target_class_name} not in db_info; skani will use defaults.")

            logging.info(f"Skani sketches in {ani_sketch_output_folder} (mode: '{skani_sketch_setting}').")
            # Corrected call to CreateANISketchFolder
            sketches_list_file = CreateANISketchFolder(
                genome_fasta_file,            # input_fasta
                ani_sketch_output_folder,     # folder_path
                arguments.threads,            # threads
                skani_sketch_setting          # mode
            )

            skani_output_file = os.path.join(target_db_base_path, "self_ANI.tsv")
            logging.info(f"Calculating Skani ANI distances (mode: '{skani_dist_setting}').")
            # Corrected call to ANIDist (using keywords as its definition has many optional args)
            ANIDist(
                ref_sketches_txt=sketches_list_file,
                query_sketches_txt=sketches_list_file,
                output=skani_output_file,
                threads=arguments.threads,
                mode=skani_dist_setting,
                force=arguments.force,
                ANI_frac_weights=use_ani_length_fraction_weight
            )
        
        elif selected_ani_tool == 'mmseqs':
            mmseqs_db_output_name = os.path.join(target_db_base_path, f"{target_class_name}_mmseqsdb_ANI")
            mmseqs_temp_dir = os.path.join(target_db_base_path, "mmseqs_tmp_ANI")
            os.makedirs(mmseqs_temp_dir, exist_ok=True)
            mmseqs_search_output_dir = os.path.join(target_db_base_path, "mmseqs_search_ANI")
            os.makedirs(mmseqs_search_output_dir, exist_ok=True)
            mmseqs_final_ani_file = os.path.join(target_db_base_path, "mmseqs_ANI.tsv")
            
            # Corrected call to MMSeqsCreateDB
            MMSeqsCreateDB(
                fasta_file=genome_fasta_file,
                db_name=mmseqs_db_output_name,
                type=2, # 2 for nucleotide
                force=arguments.force
            )
            # Corrected call to MMSeqsSearchDBs
            mmseqs_m8_file = MMSeqsSearchDBs(
                query_db=mmseqs_db_output_name,
                ref_db=mmseqs_db_output_name, # Search against itself
                outdir=mmseqs_search_output_dir,
                tmp_path=mmseqs_temp_dir,
                threads=arguments.threads,
                search_type=3, # 3 for nucleotide vs nucleotide
                force=arguments.force
            )
            # Call to m8_to_ani from ViruLink.ani.ani_calc
            m8_to_ani( # Assumes signature: m8_to_ani(m8_path, out_path, threads, len_weight=False)
                m8_path=mmseqs_m8_file,
                out_path=mmseqs_final_ani_file,
                threads=arguments.threads,
                len_weight=use_ani_length_fraction_weight
            )
        else:
            logging.warning(f"ANI tool '{selected_ani_tool}' (from YAML) not recognized for {target_class_name}. Skipping ANI generation.")

    logging_header("ViruLink Database Processing Complete")