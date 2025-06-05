import logging, sys, os
from ViruLink.utils import run_command
from glob import glob
import pandas as pd

def CreateDB(fasta_file: str, db_name: str, type: int = 0, force: bool = False):
    '''
    Create an mmseqs database from a FASTA file.
    
    Args:
        fasta_file: The path to the FASTA file.
        db_name: The name of the database to create.
        type: auto - 0, amino acid -1, nucleotides - 2
        force: Overwrite the database if it already exists.
    '''
    if not os.path.exists(db_name) or force:

        # Check if the database type is valid
        db_opts = [0, 1, 2]
        if type not in db_opts:
            logging.error(f"Invalid database type: {type}")
            sys.exit(1)
        
        cmd = f"mmseqs createdb '{fasta_file}' '{db_name}' --dbtype {type}"
        run_command(cmd)
        
        logging.info(f"Created database {db_name} from {fasta_file}")
    
    else:
        logging.info(f"Database {db_name} already exists. Use --force to overwrite.")
        

def SearchDBs(query_db, ref_db, outdir, tmp_path, threads, search_type, memory = None, force=False):
    '''
    Search the query database against the reference database using mmseqs.
    '''
    mem_string = f"--split-memory-limit {memory}" if memory else ""
    os.makedirs(outdir, exist_ok=True)
    
    if not os.path.exists(f"{outdir}/network.m8") or os.path.getsize(f"{outdir}/network.m8") == 0 or force:
        cmd = f"mmseqs search '{query_db}' '{ref_db}' '{outdir}/network_int' '{tmp_path}' -e 1.000E-05 --threads {threads} {mem_string} --search-type {search_type} -s 1"
        run_command(cmd)
        output_str = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"
        cmd2 = f"mmseqs convertalis '{query_db}' '{ref_db}' '{outdir}/network_int' '{outdir}/network.m8' --format-output {output_str} --threads {threads}"
        run_command(cmd2)
    return f"{outdir}/network.m8"

def m8_to_ANI(m8_file, output_file, force):
    if not os.path.exists(output_file) or force:
        import pandas as pd
        columns = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
                    "qstart", "qend", "tstart", "tend", "evalue", "bitscore", "qlen", "tlen"]
        
        # Read the M8 file into a DataFrame
        m8_df = pd.read_csv(m8_file, sep="\t", header=None, names=columns)
        
        # Calculate (pident * alnlen) and alignment lengths
        m8_df['pident_alnlen'] = m8_df['pident'] * m8_df['alnlen']
        
        # Group by query and target to sum necessary values
        grouped = m8_df.groupby(['query', 'target']).agg(
            sum_pident_alnlen=('pident_alnlen', 'sum'),
            sum_alnlen=('alnlen', 'sum'),
            qlen=('qlen', 'first'),
            tlen=('tlen', 'first')
        ).reset_index()
        
        # Calculate ANI using the formula and multiply by alignment length at the end
        grouped['ANI'] = ((grouped['sum_pident_alnlen'] / grouped['sum_alnlen']) / 
                          grouped[['qlen', 'tlen']].max(axis=1)) * grouped['sum_alnlen']
        
        # Save the results to the output file
        grouped[['query', 'target', 'ANI']].to_csv(output_file, index=False, sep="\t")

def DiamondCreateDB(fasta_file: str, db_name: str, force: bool = False):
    '''
    Create a DIAMOND database from a FASTA file.
    
    Args:
        fasta_file: The path to the FASTA file.
        db_name: The name of the database to create.
        force: Overwrite the database if it already exists.
    '''
    if not os.path.exists(db_name) or force:
        cmd = f"diamond makedb --in '{fasta_file}' --db '{db_name}'"
        run_command(cmd)
        logging.info(f"Created database {db_name} from {fasta_file}")
    else:
        logging.info(f"Database {db_name} already exists. Use --force to overwrite.")
        
def DiamondSearchDB(
    VOGDB_dmnd,
    query,
    outdir,
    threads,
    force=False,
    percent_id = None,
    db_cov = None):
    
    percent_id_str = "" if percent_id==None else f" --approx-id {percent_id} "
    db_cov_str = "" if db_cov==None else f" --subject-cover {db_cov} "
    
    outfile = f"{outdir}/network.m8"
    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0 or force:
        cmd = f"diamond blastx {percent_id_str} {db_cov_str} --db '{VOGDB_dmnd}' --threads {threads} --query '{query}' --out '{outfile}' --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
        run_command(cmd)
    return outfile



def m8_file_processor(
    diamond_m8_file: str,
    evalue_thresh: float,
    bitscore_thresh: float,
    output_edge_list_file: str = None,
    force_processing: bool = False):
    '''
    Process a DIAMOND m8 file to create an edge list based on e-value and bitscore thresholds.
    Args:
        diamond_m8_file: The path to the DIAMOND m8 file.
        evalue_thresh: The e-value threshold for filtering.
        bitscore_thresh: The bitscore threshold for filtering.
        output_edge_list_file: The path to the output edge list file. If None, returns a DataFrame.
        force_processing: If True, forces reprocessing of the m8 file even if the output file exists.
    Returns:
        The path to the output edge list file or a DataFrame if output_edge_list_file is None.
    '''
    print(diamond_m8_file, output_edge_list_file)
    if not output_edge_list_file==None:
        if os.path.exists(output_edge_list_file) and not force_processing:
            logging.info(f"VOG-based edge list {output_edge_list_file} already exists. Skipping m8 processing.")
            return output_edge_list_file
        
    logging.info(f"Processing m8 file {diamond_m8_file} to create edge list {output_edge_list_file}")
    cols = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]
    try:
        m8_df = pd.read_csv(diamond_m8_file, sep="\t", header=None, names=cols)
    except pd.errors.EmptyDataError:
        logging.warning(f"M8 file {diamond_m8_file} is empty. No edge list will be created.")
        open(output_edge_list_file, 'w').close() # Create empty file
        return output_edge_list_file

    filtered_m8_df = m8_df[(m8_df["evalue"] <= evalue_thresh) & (m8_df["bitscore"] >= bitscore_thresh)]
    filtered_m8_df = filtered_m8_df.sort_values(["query", "evalue", "bitscore"], ascending=[True, True, False])
    final_edge_list_df = filtered_m8_df.drop_duplicates(subset=["query", "target"], keep="first")
    
    output_df = final_edge_list_df[["query", "target", "qstart", "qend"]]
    
    if not output_edge_list_file==None:
        print("in conditional")
        output_df.to_csv(output_edge_list_file, sep="\t", index=False, header=False)
        logging.info(f"VOG-based edge list saved to {output_edge_list_file}")
        return output_edge_list_file
    else: 
        return output_df
    
    

def CreateANISketchFolder(input_fasta, folder_path, threads, mode):
    '''
    Take input fasta. Create a sketch for each sequence in the fasta within a folder
    Args:
        input_fasta: The path to the input FASTA file.
        folder_path: The path to the output folder where sketches will be saved.
        threads: The number of threads to use.
        mode: The mode to use for the sketching.
    
    Returns:
        The path to the sketches text file.
    '''
    if not os.path.exists(folder_path) or os.path.getsize(folder_path) == 0:
        cmd = f"skani sketch -i '{input_fasta}' -o '{folder_path}' {mode} -t {threads}"
        run_command(cmd)
    sketches = glob(f"{folder_path}/*.sketch")
    txt_content = '\n'.join(sketches)
    txt_path = f"{folder_path}/sketches.txt"
    with open(txt_path, "w") as f:
        f.write(txt_content)
    return txt_path

def FixANIOutput(ANI_output, ANI_frac_weights=False):
    import pandas as pd
    # Load safely
    ANI_edges = pd.read_csv(ANI_output, sep="\t", low_memory=False)

    # Check if headers exist and rename columns accordingly
    if "Ref_file" in ANI_edges.columns:
        rename_dict = {
            "Query_name": "source",
            "Ref_name": "target",
        }
        ANI_edges = ANI_edges.rename(columns=rename_dict)

        # Make sure source and target are strings before split
        ANI_edges["source"] = ANI_edges["source"].astype(str).map(lambda x: x.split(' ')[0])
        ANI_edges["target"] = ANI_edges["target"].astype(str).map(lambda x: x.split(' ')[0])

        # Compute weighted ANI
        if ANI_frac_weights:
            ANI_edges["weight"] = ANI_edges["ANI"] * ANI_edges[["Align_fraction_query", "Align_fraction_ref"]].max(axis=1) / 10000
        else:
            ANI_edges["weight"] = ANI_edges["ANI"]
        # Drop unwanted columns
        cols_to_drop = ["Ref_file", "Query_file", "Align_fraction_query", "Align_fraction_ref", "ANI"]
        ANI_edges = ANI_edges.drop(columns=cols_to_drop, errors="ignore")
        
        ANI_edges.to_csv(ANI_output, sep="\t", index=False)
    #draw_taxonomic_subgraphs(ANI_edges, 5000, seed=42)
    return ANI_edges


def ANIDist(ref_sketches_txt, query_sketches_txt, output, threads, mode, force=False, ANI_frac_weights=False):
    '''
    Calculate ANI distances between two sets of sketches.
    
    Args:
        ref_sketches_txt: The path to the reference sketches text file.
        query_sketches_txt: The path to the query sketches text file.
        output: The path to the output file where distances will be saved.
        threads: The number of threads to use.
        mode: The mode to use for the distance calculation.
        force: Overwrite the output file if it already exists.
    '''
    if not os.path.exists(output) or os.path.getsize(output) == 0 or force:
        if not os.path.exists(ref_sketches_txt):
            logging.error(f"Reference sketches file {ref_sketches_txt} does not exist.")
            sys.exit(1)
        if not os.path.exists(query_sketches_txt):
            logging.error(f"Query sketches file {query_sketches_txt} does not exist.")
            sys.exit(1)
        cmd = f"skani dist --rl '{ref_sketches_txt}' --ql '{query_sketches_txt}' -o '{output}' -t {threads} {mode}"
        run_command(cmd)
        
    ANI_edges = FixANIOutput(output, ANI_frac_weights)
    return ANI_edges
    