import logging, sys, os
from ViraLink.utils import run_command

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
        

def SearchDBs(query_db, ref_db, outdir, tmp_path, threads, memory):
    '''
    Search the query database against the reference database using mmseqs.
    '''
    if not os.path.exists(f"{outdir}/network.m8"):
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

