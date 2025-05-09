from ViruLink.search_utils import CreateDB, SearchDBs
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
    database_exceptions = {
        "VOGDB": 1
    }
    db_outpath = f"{unproc_path}/{class_data}_mmseqsDB"
    
    if class_data in database_exceptions:
        dbtype=database_exceptions[class_data]
    else:
        dbtype=2
    
    CreateDB(get_seqs_path(unproc_path), db_outpath, type=dbtype)
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
        
        # Create mmseqs databases
        mmseqs_db_paths = {}
        for class_data, unproc_path in paths_to_unprocs.items():
            logging.info(f"Processing {class_data} unprocessed data.")
            mmseqs_db_paths[class_data] = generate_database(class_data, unproc_path)

        # VOGDB is treated specially
        VOGDB_mmseqsDB = mmseqs_db_paths["VOGDB"]
        del mmseqs_db_paths["VOGDB"]
        
        # Search and create m8 files
        m8_files = {}
        for class_data, mmseqs_db_path in mmseqs_db_paths.items():
            
            process_db = f"{arguments.databases_loc}/{class_data}" #processed in same folder
            process_tmp = f"{arguments.databases_loc}/{class_data}/tmp"
            os.makedirs(process_db, exist_ok=True)
            os.makedirs(process_tmp, exist_ok=True)
            
            logging.info(f"Searching {class_data} against VOGDB.")
            m8_files[class_data] = SearchDBs(
                mmseqs_db_path,
                VOGDB_mmseqsDB,
                process_db,
                process_tmp,
                arguments.threads,
                arguments.memory
            )

        # process m8 files and build adjacency
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
            
            
            

            
            
            

            '''
            from gensim.models import Word2Vec
            walk_length = 20
            p = 0.5
            q = 1.0
            num_threads = 32
            walks_per_node = 1
            embedding_dim = 64
            
            from ViraLink.utils import (
                prepare_edges_for_cpp,
                make_all_nodes_list,
                run_biased_random_walk
            )
            
            row_int, col_int, weights_float, label_to_id, id_to_label = prepare_edges_for_cpp(
                sources, destinations, weights
            )

            from ViraLink.random_walk import biased_random_walk
            walks = biased_random_walk.random_walk(
                row_int, col_int, 
                make_all_nodes_list(label_to_id), 
                weights_float, 
                walk_length, 
                walks_per_node=100, 
                num_threads=32, 
                p=1, 
                q=1.0
            )
            
            # Then train node2vec, etc.
            walks_str = [[str(nid) for nid in walk] for walk in walks]
            model = Word2Vec(
                sentences=walks_str,
                vector_size=embedding_dim,
                window=5,
                min_count=0,
                sg=1,         # skip-gram
                workers=num_threads,
                epochs=5
            )

            # Extract final embeddings
            emb_dict = {}
            for node_id, lbl in id_to_label.items():
                key = str(node_id)
                if key in model.wv:
                    emb_dict[lbl] = model.wv[key]
                else:
                    emb_dict[lbl] = None

            # Step 5: Plot t-SNE
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import numpy as np
            
            def plot_umap_embeddings(emb_dict, meta, output_path="umap.html"):
                """
                Plot a UMAP visualization of node embeddings using Plotly.
                Points are colored by Genus. Hovering over a point shows Species, Family, and Genus.
                
                Parameters
                ----------
                emb_dict : dict
                    Mapping from node_label -> embedding (np.ndarray). 
                    Example: emb_dict["YP_009123.1"] = np.array([...])
                meta : pd.DataFrame
                    Must contain columns "Accession", "Species", "Family", "Genus".
                    For example:
                        Accession      Species        Family           Genus
                        ---------      -------        ------           -----
                        YP_009123      SomeSpecies    SomeFamily       SomeGenus
                output_path : str, optional
                    File path for the saved HTML plot (default is "umap.html").
                """
                import pandas as pd
                import numpy as np
                from umap import UMAP
                import plotly.express as px

                # Build a mapping from Accession to its metadata for quick look-up.
                accession_to_species = dict(zip(meta["Accession"], meta["Species"]))
                accession_to_family  = dict(zip(meta["Accession"], meta["Family"]))
                accession_to_genus   = dict(zip(meta["Accession"], meta["Genus"]))
                
                # Collect embeddings and corresponding metadata.
                points = []
                accessions = []
                species_list = []
                family_list = []
                genus_list = []
                
                for label, emb in emb_dict.items():
                    if emb is None:
                        continue  # Skip nodes with no embedding.
                    
                    # Retrieve metadata. If genus is invalid or "Unknown", skip the node.
                    genus = accession_to_genus.get(label, "Unknown")
                    if pd.isna(genus) or not isinstance(genus, str) or genus.strip() == "" or genus == "Unknown":
                        continue

                    species = accession_to_species.get(label, "Unknown")
                    if pd.isna(species) or not isinstance(species, str) or species.strip() == "":
                        species = "Unknown"

                    family = accession_to_family.get(label, "Unknown")
                    if pd.isna(family) or not isinstance(family, str) or family.strip() == "":
                        family = "Unknown"

                    points.append(emb)
                    accessions.append(label)
                    species_list.append(species)
                    family_list.append(family)
                    genus_list.append(genus)
                
                X = np.array(points)
                if X.shape[0] < 2:
                    print("Not enough embeddings to plot UMAP (need at least 2).")
                    return

                # Run UMAP on the embeddings.
                umap = UMAP(n_components=2, random_state=42)
                X_2d = umap.fit_transform(X)
                
                # Build a DataFrame for Plotly with the UMAP coordinates and metadata.
                df_umap = pd.DataFrame({
                    "x": X_2d[:, 0],
                    "y": X_2d[:, 1],
                    "Accession": accessions,
                    "Species": species_list,
                    "Family": family_list,
                    "Genus": genus_list
                })

                # Create an interactive scatter plot, coloring points by Genus.
                # Hover data will display Species, Family, and Genus.
                fig = px.scatter(
                    df_umap,
                    x="x",
                    y="y",
                    color="Family",          # Use Genus to set the color.
                    hover_data=["Species", "Family", "Genus"]
                )
                
                # Remove the legend.
                fig.update_layout(showlegend=False)
                
                # Optionally, tweak marker size and opacity:
                fig.update_traces(marker=dict(size=8, opacity=0.8))
                
                # Save the plot as an HTML file.
                fig.write_html(output_path)
                print(f"UMAP interactive plot saved to {output_path}")


                
            plot_umap_embeddings(emb_dict, meta)'''