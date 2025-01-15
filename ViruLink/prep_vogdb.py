import logging
import os
import subprocess
import pandas as pd
import numpy as np
from math import log10, ceil
import multiprocessing as mp
from itertools import zip_longest, chain
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from scipy.stats import hypergeom
from scipy.sparse import csr_matrix
from Bio.Phylo.TreeConstruction import DistanceCalculator
from ViraLink.utils import running_message, run_command, read_fasta


@running_message
def prep_vogdb(vogdb_path: str):
    '''
    Prepare the mmseqs database of VOGs.
    '''
    vogdb_dir = os.path.dirname(vogdb_path)
    mmseqs_db_path = os.path.join(vogdb_dir, "vogdb_mmseqs")

    if os.path.exists(mmseqs_db_path) and os.path.getsize(mmseqs_db_path) > 0:
        logging.info(f"VOG database already prepared at: {mmseqs_db_path}")
        return mmseqs_db_path

    logging.info("Preparing the VOG database using mmseqs.")
    os.makedirs(vogdb_dir, exist_ok=True)
    try:
        cmd = f"mmseqs createdb {vogdb_path} {mmseqs_db_path} --dbtype 1"
        run_command(cmd)
        logging.info(f"VOG database prepared at: {mmseqs_db_path}")
    except Exception as e:
        logging.error(f"Failed to prepare the VOG database:")
        raise e
    return mmseqs_db_path


@running_message
def prep_nucleotide_db(fasta_path: str, output_dir: str):
    '''
    Prepare the mmseqs database for nucleotide sequences inside the output directory.
    '''
    mmseqs_db_path = os.path.join(output_dir, "nucleotide_mmseqs")

    if os.path.exists(mmseqs_db_path) and os.path.getsize(mmseqs_db_path) > 0:
        logging.info(f"Nucleotide database already prepared at: {mmseqs_db_path}")
        return mmseqs_db_path

    logging.info("Preparing the nucleotide database using mmseqs.")
    os.makedirs(output_dir, exist_ok=True)
    try:
        cmd = f"mmseqs createdb {fasta_path} {mmseqs_db_path} --dbtype 2"
        run_command(cmd)
        logging.info(f"Nucleotide database prepared at: {mmseqs_db_path}")
    except Exception as e:
        logging.error("Failed to prepare the nucleotide database:")
        raise e

    return mmseqs_db_path


@running_message
def mmseqs_search(query_db, ref_db, outdir, tmp_path, threads, memory):
    '''
    Search the query database against the reference database using mmseqs.
    '''
    if not os.path.exists(f"{outdir}/network.m8"):
        cmd = f"mmseqs search {query_db} {ref_db} {outdir}/network_int {tmp_path} --threads {threads} --split-memory-limit {memory}"
        run_command(cmd, shell=True, check=True)
        cmd2 = f"mmseqs convertalis {query_db} {ref_db} {outdir}/network_int {outdir}/network.m8"
        run_command(cmd2, shell=True, check=True)
    return f"{outdir}/network.m8"


def select_hallmark_genes(presence_absence, top_n=3):
    '''
    Select hallmark genes from the presence-absence table by picking the top N most common queries in each iteration.
    '''
    pa_table = presence_absence.copy()
    hallmark_genes = []

    while not pa_table.empty:
        query_sums = pa_table.sum(axis=0)

        top_queries = query_sums.nlargest(top_n).index.tolist()
        hallmark_genes.extend(top_queries)

        subjects_with_queries = pa_table.index[pa_table[top_queries].any(axis=1)]

        pa_table.drop(index=subjects_with_queries, inplace=True)

        pa_table.drop(columns=top_queries, inplace=True)

    return hallmark_genes


def extract_and_align_hallmark_genes(
    hallmark_genes: list,
    search_df: pd.DataFrame,
    seqs_path: str,
    output_dir: str,
    threads: int
):
    '''
    For each hallmark gene, extract the sequences, align them using MAFFT via command line, and generate a distance matrix.
    '''
    logging.info("Extracting and aligning hallmark genes")

    # Create directories
    fasta_dir = os.path.join(output_dir, "hallmark_fastas")
    align_dir = os.path.join(output_dir, "hallmark_alignments")
    distance_dir = os.path.join(output_dir, "hallmark_distances")

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(distance_dir, exist_ok=True)

    # Read the nucleotide sequences
    sequences = {record.id: record for record in read_fasta(seqs_path)}

    for gene in hallmark_genes:
        distance_file = os.path.join(distance_dir, f"{gene}_distance.tsv")
        if os.path.exists(distance_file):
            logging.info(f"Distance matrix for gene {gene} already exists. Skipping computation.")
            continue  # Skip to the next gene

        logging.info(f"Processing hallmark gene: {gene}")

        # Get the hits for this gene
        gene_hits = search_df[search_df['query'] == gene]

        # Extract sequences for subjects (targets)
        records = []
        for idx, row in gene_hits.iterrows():
            subject_id = row['target']
            qstart = int(row['qstart']) - 1
            qend = int(row['qend'])
            seq_record = sequences.get(subject_id)
            if seq_record:
                # Extract the sequence region
                seq_fragment = seq_record.seq[qstart:qend]
                new_record = SeqRecord(
                    seq=seq_fragment,
                    id=subject_id,
                    description=''
                )
                records.append(new_record)
            else:
                logging.warning(f"Sequence for subject {subject_id} not found in input sequences.")

        if not records:
            logging.warning(f"No sequences found for hallmark gene {gene}. Skipping.")
            continue

        # Save the extracted sequences to a FASTA file
        fasta_file = os.path.join(fasta_dir, f"{gene}.fasta")
        with open(fasta_file, 'w') as f_out:
            SeqIO.write(records, f_out, 'fasta')

        # Align the sequences using MAFFT
        alignment_file = os.path.join(align_dir, f"{gene}_aligned.fasta")
        cmd = f"mafft --quiet --auto --thread {threads} {fasta_file}"

        try:
            # Run MAFFT and capture the output
            result = run_command(cmd)
            # Write the alignment
            with open(alignment_file, 'w') as f_out:
                f_out.write(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"MAFFT failed for gene {gene}: {e}")
            continue

        # Generate distance matrix
        alignment = AlignIO.read(alignment_file, 'fasta')
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)

        # Save the distance matrix
        with open(distance_file, 'w') as f_out:
            f_out.write('\t' + '\t'.join(dm.names) + '\n')
            for name1 in dm.names:
                distances = [f"{dm[name1, name2]:.5f}" for name2 in dm.names]
                f_out.write(name1 + '\t' + '\t'.join(distances) + '\n')

        logging.info(f"Processed hallmark gene: {gene}")


def combine_distance_matrices(
    distance_dir: str,
    output_dir: str
):
    '''
    Combine all individual distance matrices into a single comprehensive weighted adjacency matrix.
    '''
    import os
    import pandas as pd
    import numpy as np
    from functools import reduce

    combined_distance_file = os.path.join(output_dir, 'combined_distance_matrix.tsv')
    weighted_adjacency_matrix_file = os.path.join(output_dir, 'weighted_adjacency_matrix.tsv')

    if os.path.exists(weighted_adjacency_matrix_file):
        logging.info("Combined weighted adjacency matrix already exists. Skipping recomputation.")
        return

    logging.info("Combining individual distance matrices into a comprehensive weighted adjacency matrix")

    # List all distance matrix files
    distance_files = [os.path.join(distance_dir, f) for f in os.listdir(distance_dir) if f.endswith('_distance.tsv')]

    if not distance_files:
        logging.warning("No distance matrices found to combine.")
        return

    # Initialize a list to hold DataFrames
    distance_dfs = []

    for file in distance_files:
        # Read the distance matrix into a DataFrame
        df = pd.read_csv(file, sep='\t', index_col=0)
        distance_dfs.append(df)

    # Align all DataFrames on rows and columns
    sum_distances = reduce(lambda x, y: x.add(y, fill_value=0), distance_dfs)
    counts = reduce(lambda x, y: x.add(~y.isna(), fill_value=0), distance_dfs)

    # avoid division by zero
    counts.replace(0, np.nan, inplace=True)

    # Compute average distances
    average_distance = sum_distances.divide(counts)

    # Replace NaN values
    average_distance = average_distance.fillna(1.0)

    average_distance.to_csv(combined_distance_file, sep='\t')
    logging.info(f"Combined distance matrix saved to {combined_distance_file}")

    weights = 1 - average_distance
    weights = weights.clip(lower=0.0, upper=1.0)

    # no self loops
    np.fill_diagonal(weights.values, 0)

    weights.to_csv(weighted_adjacency_matrix_file, sep='\t')
    logging.info(f"Combined weighted adjacency matrix saved to {weighted_adjacency_matrix_file}")



def calculate_hypergeometric_adjacency(presence_absence: pd.DataFrame, 
                                       output_dir: str, 
                                       p_value_threshold: float = 0.05,
                                       min_gene_count: int = 1):
    adjacency_matrix_file = os.path.join(output_dir, 'hypergeometric_adjacency_matrix.tsv')

    if os.path.exists(adjacency_matrix_file):
        logging.info("Hypergeometric adjacency matrix already exists. Skipping recomputation.")
        return

    logging.info("Calculating hypergeometric adjacency matrix from presence-absence matrix.")

    # Optional: Filter out subjects with very low gene counts (if desired)
    # This reduces the size of the problem.
    subject_gene_counts = presence_absence.sum(axis=1)
    filtered_presence_absence = presence_absence.loc[subject_gene_counts >= min_gene_count]
    if len(filtered_presence_absence) < len(presence_absence):
        logging.info(f"Filtered subjects from {len(presence_absence)} to {len(filtered_presence_absence)} based on min_gene_count={min_gene_count}.")
    presence_absence = filtered_presence_absence

    M = presence_absence.shape[1]  # Total number of genes
    gene_counts = presence_absence.sum(axis=1).values
    subjects = presence_absence.index
    num_subjects = len(subjects)

    # Convert to sparse for efficient multiplication if matrix is large and sparse
    presence_absence_sparse = csr_matrix(presence_absence.values)

    # Compute shared gene counts using sparse multiplication
    # shared_gene_counts will be a dense array after conversion
    shared_gene_counts = (presence_absence_sparse.dot(presence_absence_sparse.T)).toarray()

    # Get upper triangular indices
    triu_indices = np.triu_indices(num_subjects, k=1)
    i_idx, j_idx = triu_indices

    N = gene_counts[i_idx]  # gene count for subject i
    n = gene_counts[j_idx]  # gene count for subject j
    k = shared_gene_counts[i_idx, j_idx]

    # Pre-allocate p_vals array for efficiency
    p_vals = np.ones_like(k, dtype=float)

    # Compute p-values, but skip cases that are trivially not significant
    # For instance, if k == 0, p-value = 1, no need to compute hypergeom.
    nonzero_mask = (k > 0)
    # Compute only for nonzero k
    p_vals[nonzero_mask] = hypergeom.sf(k[nonzero_mask] - 1, M, N[nonzero_mask], n[nonzero_mask])

    # Apply significance threshold
    significant_mask = (p_vals <= p_value_threshold)
    # Add a small epsilon before taking the log
    epsilon = 1e-15
    weights_upper = np.zeros_like(p_vals)
    weights_upper[significant_mask] = -np.log(p_vals[significant_mask] + epsilon)

    # Construct the full adjacency (weight) matrix
    weights = np.zeros((num_subjects, num_subjects))
    weights[i_idx, j_idx] = weights_upper
    weights[j_idx, i_idx] = weights_upper  # symmetric

    # No self loops
    np.fill_diagonal(weights, 0)

    adjacency_df = pd.DataFrame(weights, index=subjects, columns=subjects)
    adjacency_df.to_csv(adjacency_matrix_file, sep='\t')
    logging.info(f"Hypergeometric weighted adjacency matrix saved to {adjacency_matrix_file}")

def Prepare_data(
    seqs: str,
    map_bitscore_threshold: int,
    reference_data: str,
    output: str,
    threads: int,
    memory: str,
    p_value_threshold: float = 0.05,
    min_occurrence: int = 2
):
    '''
    Prepare the data for training and generate necessary matrices.
    
    Parameters
    ----------
    seqs : str
        Path to the input nucleotide sequences.
    map_bitscore_threshold : int
        Bitscore threshold for filtering mmseqs search hits.
    reference_data : str
        Path to the directory containing reference VOG database files.
    output : str
        Output directory where results will be stored.
    threads : int
        Number of threads for parallel operations.
    memory : str
        Memory limit for mmseqs (e.g., "4G").
    p_value_threshold : float, optional
        p-value threshold for the hypergeometric test, by default 0.05.
    min_occurrence : int, optional
        Minimum number of genomes a gene must be present in to be included.
        - If set to 1, genes present in no genomes (sum=0) are excluded.
        - If set to 2, genes present in only one genome (sum=1) are also excluded.
        Default is 1.
    '''
    logging.info("Preparing the data")
    vogdb_faa_path = f"{reference_data}/vogdb_merged.faa"

    # Prepare mmseqs databases
    vog_mmseqs_db = prep_vogdb(vogdb_faa_path)
    seq_mmseqs_db = prep_nucleotide_db(seqs, output)

    # Run mmseqs search
    search_output_dir = f"{output}/search_output"
    os.makedirs(search_output_dir, exist_ok=True)

    tmp = f"{output}/tmp"
    os.makedirs(tmp, exist_ok=True)

    search_output = mmseqs_search(vog_mmseqs_db, seq_mmseqs_db, search_output_dir, tmp, threads, memory)
    columns = ["query", "target", "pident", "alnlen", "mismatch", "numgapopen",
               "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]
    search_df = pd.read_csv(search_output, sep='\t', header=None, names=columns)

    pres_abs_path = f"{output}/presence_absence.tsv"

    # Filter hits by bitscore threshold and keep best hits per (query, target) pair
    filtered_df = search_df[search_df["bitscore"] >= map_bitscore_threshold]
    filtered_df = filtered_df.sort_values(by=["query", "target", "bitscore"], ascending=[True, True, False])
    filtered_df = filtered_df.drop_duplicates(subset=["query", "target"], keep="first")

    # Generate the presence-absence table
    presence_absence = pd.crosstab(filtered_df["target"], filtered_df["query"])

    # Exclude genes (columns) not meeting the minimum occurrence criterion
    # Calculate column sums (each column is a gene presence vector)
    col_sums = presence_absence.sum(axis=0)
    # Keep only those columns where sum >= min_occurrence
    presence_absence = presence_absence.loc[:, col_sums >= min_occurrence]

    # If min_occurrence=1, this removes genes with sum=0 (no presence).
    # If min_occurrence=2, this also removes genes with sum=1 (singletons).

    presence_absence.to_csv(pres_abs_path, sep='\t')
    logging.info(f"Presence-absence table saved to {pres_abs_path}")

    hallmark_genes_file = f"{output}/hallmark_genes.txt"
    if not os.path.exists(hallmark_genes_file):
        hallmark_genes = select_hallmark_genes(presence_absence)
        with open(hallmark_genes_file, 'w') as f:
            for gene in hallmark_genes:
                f.write(f"{gene}\n")
        logging.info(f"Hallmark genes selected and saved to {hallmark_genes_file}")
    else:
        with open(hallmark_genes_file, 'r') as f:
            hallmark_genes = [line.strip() for line in f]
        logging.info(f"Hallmark genes loaded from {hallmark_genes_file}")

    # Extract sequences for each hallmark gene
    extract_and_align_hallmark_genes(
        hallmark_genes,
        filtered_df,
        seqs,
        output,
        threads
    )

    # Combine all distance matrices into a weighted adjacency matrix
    distance_dir = os.path.join(output, "hallmark_distances")
    combine_distance_matrices(
        distance_dir=distance_dir,
        output_dir=output
    )

    # Calculate binary hypergeometric adjacency matrix
    calculate_hypergeometric_adjacency(
        presence_absence=presence_absence,
        output_dir=output,
        p_value_threshold=p_value_threshold
    ) 