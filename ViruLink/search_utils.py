import logging, sys, os
from ViruLink.utils import run_command
from glob import glob

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
        
        cmd = f"mmseqs createdb {fasta_file} {db_name} --dbtype {type}"
        run_command(cmd)
        
        logging.info(f"Created database {db_name} from {fasta_file}")
    
    else:
        logging.info(f"Database {db_name} already exists. Use --force to overwrite.")
        

def SearchDBs(query_db, ref_db, outdir, tmp_path, threads, memory, force=False):
    '''
    Search the query database against the reference database using mmseqs.
    '''
    if not os.path.exists(f"{outdir}/network.m8") or os.path.getsize(f"{outdir}/network.m8") == 0 or force:
        cmd = f"mmseqs search {query_db} {ref_db} {outdir}/network_int {tmp_path} --threads {threads} --split-memory-limit {memory}"
        run_command(cmd)
        cmd2 = f"mmseqs convertalis {query_db} {ref_db} {outdir}/network_int {outdir}/network.m8"
        run_command(cmd2)
    return f"{outdir}/network.m8"


def DiamondCreateDB(fasta_file: str, db_name: str, force: bool = False):
    '''
    Create a DIAMOND database from a FASTA file.
    
    Args:
        fasta_file: The path to the FASTA file.
        db_name: The name of the database to create.
        force: Overwrite the database if it already exists.
    '''
    if not os.path.exists(db_name) or force:
        cmd = f"diamond makedb --in {fasta_file} --db {db_name}"
        run_command(cmd)
        logging.info(f"Created database {db_name} from {fasta_file}")
    else:
        logging.info(f"Database {db_name} already exists. Use --force to overwrite.")
        
def DiamondSearchDB(VOGDB_dmnd, query, outdir, threads):
    outfile = f"{outdir}/network.m8"
    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        cmd = f"diamond blastx --fast --db {VOGDB_dmnd} --threads {threads} --query {query} --out {outfile} --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
        run_command(cmd)
    return outfile

def CreateANISketchFolder(input_fasta, folder_path, threads, mode, force=False, ):
    '''
    Take input fasta. Create a sketch for each sequence in the fasta within a folder
    Args:
        input_fasta: The path to the input FASTA file.
        folder_path: The path to the output folder where sketches will be saved.
        force: Overwrite the sketches if they already exist.
    '''
    if not os.path.exists(folder_path) or force or os.path.getsize(folder_path) == 0:
        cmd = f"skani sketch -i {input_fasta} -o {folder_path} {mode} -t {threads}"
        run_command(cmd)
    sketches = glob(f"{folder_path}/*.sketch")
    txt_content = '\n'.join(sketches)
    txt_path = f"{folder_path}/sketches.txt"
    with open(txt_path, "w") as f:
        f.write(txt_content)
    return txt_path

def FixANIOutput(ANI_output):
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
        ANI_edges["weight"] = ANI_edges["ANI"] * ANI_edges[["Align_fraction_query", "Align_fraction_ref"]].max(axis=1) / 10000
        #ANI_edges["weight"] = ANI_edges["ANI"]
        # Drop unwanted columns
        cols_to_drop = ["Ref_file", "Query_file", "Align_fraction_query", "Align_fraction_ref", "ANI"]
        ANI_edges = ANI_edges.drop(columns=cols_to_drop, errors="ignore")
        
        ANI_edges.to_csv(ANI_output, sep="\t", index=False)
    #draw_taxonomic_subgraphs(ANI_edges, 5000, seed=42)
    return ANI_edges


def ANIDist(ref_sketches_txt, query_sketches_txt, output, threads, mode, force=False):
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
        cmd = f"skani dist --rl {ref_sketches_txt} --ql {query_sketches_txt} -o {output} -t {threads} {mode}"
        run_command(cmd)
        
    ANI_edges = FixANIOutput(output)
    return ANI_edges
    