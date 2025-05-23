import os, logging, sys, shlex, subprocess
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from scipy.special import ndtr
from matplotlib import pyplot as plt


def get_unique_log_filename(base_log_filename):
    """
    If the log file already exists, append a number to the filename to make it unique.
    """
    
    i = 1
    base_name, extension = os.path.splitext(base_log_filename)
    while os.path.exists(base_log_filename):
        base_log_filename = f"{base_name}_{i}{extension}"
        i += 1
    return base_log_filename

def init_logging(log_filename: str):
    '''
    Initialize logging to a file and stdout.
    
    Args:
        log_filename: The name of the log file.
    '''
    class TeeHandler(logging.Handler):
        def __init__(self, filename, mode='a'):
            super().__init__()
            self.file = open(filename, mode)
            self.stream_handler = logging.StreamHandler(sys.stdout)

        def emit(self, record):
            log_entry = self.format(record)
            self.file.write(log_entry + '\n')
            self.file.flush()
            self.stream_handler.emit(record)

        def close(self):
            self.file.close()
            super().close()
    i=1
    while os.path.exists(log_filename):
        log_filename = get_unique_log_filename(log_filename)
    print(f"Logging to {log_filename}")

    tee_handler = TeeHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    tee_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[tee_handler])

def run_command(cmd, **kwargs):
    """
    Run *cmd* and stream its stdout/stderr in real-time to both the console
    and the log file handled by the logging system.

    Returns
    -------
    subprocess.CompletedProcess
    """
    # ---------- default Popen flags --------------------------------------
    kw = kwargs.copy()
    kw.setdefault("text", True)            # decode bytes → str
    kw.setdefault("bufsize", 1)            # line-buffered
    kw.setdefault("stdout", subprocess.PIPE)
    kw.setdefault("stderr", subprocess.STDOUT)  # merge both streams

    # ---------- command list ---------------------------------------------
    if kw.get("shell", False):
        cmd_list = cmd                     # user handles quoting
    else:
        cmd_list = shlex.split(cmd)

    logging.info(f"Running command: {cmd}")

    # ---------- launch & stream ------------------------------------------
    proc = subprocess.Popen(cmd_list, **kw)
    captured = []                          # keep lines if caller wants them

    try:
        for line in proc.stdout:           # iterates line-by-line
            line = line.rstrip("\n")
            captured.append(line)
            logging.info(line)             # TeeHandler writes to file + console

        proc.wait()                        # ensure the process is done
    except Exception:                      # propagate logging on Ctrl-C, etc.
        proc.kill()
        proc.wait()
        raise

    # ---------- error handling -------------------------------------------
    if proc.returncode != 0:
        logging.error(f"Command '{cmd}' failed with return code {proc.returncode}")
        raise subprocess.CalledProcessError(proc.returncode, cmd,
                                            output="\n".join(captured))

    logging.info(f"Command '{cmd}' completed successfully")

    return subprocess.CompletedProcess(cmd, proc.returncode,
                                       "\n".join(captured), "")


def running_message(function):
    def wrapper(*args, **kwargs):
        def format_argument(arg):
            if isinstance(arg, pd.DataFrame):
                return f"DataFrame({len(arg)} rows x {len(arg.columns)} columns)"
            elif isinstance(arg, (list, dict)) and len(arg) > 10:
                return f"{type(arg).__name__}({len(arg)} items)"
            return repr(arg)

        def format_timedelta(delta):
            seconds = delta.total_seconds()
            if seconds < 60:
                return f"{seconds:.2f} seconds"
            elif seconds < 3600:
                minutes = seconds / 60
                return f"{minutes:.2f} minutes"
            elif seconds < 86400:
                hours = seconds / 3600
                return f"{hours:.2f} hours"
            else:
                days = seconds / 86400
                return f"{days:.2f} days"

        T1 = datetime.now()
        current_time = T1.strftime("%H:%M:%S")
        arg_names = function.__code__.co_varnames[:function.__code__.co_argcount]
        args_repr = [f"{arg}={format_argument(a)}" for arg, a in zip(arg_names, args)]
        kwargs_repr = [f"{k}={format_argument(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logging.info(f"Time: {current_time} - Running {function.__name__} with inputs: {function.__name__}({signature})")

        try:
            result = function(*args, **kwargs)
            return result
        except Exception as e:
            logging.exception(f"Exception occurred in function {function.__name__}: {e}")
            raise
        finally:
            T2 = datetime.now()
            current_time2 = T2.strftime("%H:%M:%S")
            total_time = format_timedelta(T2 - T1)
            if 'verify_output' in kwargs:
                if os.stat(kwargs['verify_output'])== 0:
                    logging.error(f"Time: {current_time2} - {function.__name__} Failed")
                    logging.error(f"Total time taken: {total_time}")
            logging.info(f"Time: {current_time2} - {function.__name__} Completed")
            logging.info(f"Total time taken: {total_time}")

    return wrapper



def read_fasta(fastafile):
    # Get the total size of the file
    total_size = os.path.getsize(fastafile)
    
    # Get the total number of records to parse with a progress bar for reading lines
    with open(fastafile) as f, tqdm(total=total_size, desc="Reading FASTA file", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        total_records = 0
        for line in f:
            pbar.update(len(line))
            if line.startswith(">"):
                total_records += 1
    
    # Parse the FASTA file with a progress bar
    with tqdm(total=total_records, desc="Parsing FASTA file", unit=" Records") as pbar:
        records = []
        for record in SeqIO.parse(fastafile, "fasta"):
            records.append(record)
            pbar.update(1)
    
    return records

def write_fasta(sequences, outpath):
    with open(outpath, "w") as fasta_out:
        SeqIO.write(sequences, fasta_out, "fasta")

def get_file_path(unproc_path, ext, multi=False):
    from glob import glob
    all_paths = glob(f"{unproc_path}/*.{ext}")
    if not multi:
        if len(all_paths) == 0:
            logging.error(f"No {ext} files found in {unproc_path}")
            sys.exit(1)
        elif len(all_paths) > 1:
            logging.error(f"Multiple {ext} files found in {unproc_path}")
            sys.exit(1)
        else:
            return all_paths[0]
    return all_paths

def edge_list_to_presence_absence(edge_list_path):
    """
    Converts an edge list into a presence-absence matrix.
    
    Parameters:
        edge_list_path (str): Path to the edge list file. The file should have two columns: query and target.
        
    Returns:
        pd.DataFrame: A presence-absence matrix where rows are queries, columns are targets,
                      and values are 1 (presence) or 0 (absence).
    """
    # Read the edge list
    edge_list = pd.read_csv(edge_list_path, sep="\t", header=None, names=["query", "target", "qstart", "qend"])
    edge_list = edge_list[["query", "target"]]
    
    # Create a presence-absence matrix using pandas pivot_table
    presence_absence_matrix = (edge_list
                               .assign(presence=1)  # Add a column with value 1 to indicate presence
                               .pivot_table(index="query", columns="target", values="presence", fill_value=0)
                              )
    
    # Optional: Reset the column names for cleaner formatting
    presence_absence_matrix.columns.name = None
    presence_absence_matrix.index.name = None

    return presence_absence_matrix





def compute_hypergeom_weights(
        pa_matrix: pd.DataFrame,
        nthreads: int,
        pval_thresh: float = 0.1,
        max_freq: float = 0.80,
        hypergeom: bool = False          # NEW
) -> pd.DataFrame:
    """
    Compute a genome-pair weight matrix from a presence/absence table
    using a one-sided hyper-geometric tail test.

    Parameters
    ----------
    pa_matrix : DataFrame [G × P] (bool / 0-1)
        Presence/absence matrix: rows = genomes, columns = proteins.
    nthreads : int, default 1
        OpenMP thread count for the C++ kernel.
    pval_thresh : float, default 0.1
        Significance level α for the one-sided test.
    max_freq : float, default 0.80
        Discard proteins present in > max_freq × G genomes.
    hypergeom : bool, default False
        • False → weight = c / min(k_i, k_j)  (0,1]   (original behaviour)  
        • True  → weight = –log10(p-value)          (0 on non-significant pairs)

    Returns
    -------
    DataFrame [G × G] (float)
        Symmetric weight matrix.
    """
    # --- call the C++ kernel --------------------------------------------
    from ViruLink.hypergeom import hypergeom as _hyper_mod

    bool_array = np.ascontiguousarray(pa_matrix.values.astype(bool))

    w_mat = _hyper_mod.compute_hypergeom(
        bool_array,
        nthreads=nthreads,
        pval_thresh=pval_thresh,
        max_freq=max_freq,
        return_log=hypergeom         # NEW
    )

    # --- diagnostics -----------------------------------------------------
    total_pairs   = (w_mat.size - w_mat.shape[0]) // 2
    nonzero_pairs = int((w_mat > 0).sum() // 2)
    zero_pairs    = total_pairs - nonzero_pairs

    print("[hypergeom] total genome pairs        :", total_pairs)
    print("[hypergeom] pairs with non-zero weight:", nonzero_pairs)
    print("[hypergeom] pairs with zero weight    :", zero_pairs)

    # --- wrap in DataFrame ----------------------------------------------
    idx = pa_matrix.index
    return pd.DataFrame(w_mat, index=idx, columns=idx)


def create_graph(pval_df: pd.DataFrame, threshold: float = 0.1) -> tuple:
    """
    Vectorized conversion of a DataFrame of -log(p-values) into edge lists for graph construction.

    Parameters
    ----------
    pval_df : pd.DataFrame
        A square DataFrame where rows and columns represent nodes, and values
        are the edge weights (-log(p-values)).
    threshold : float, optional
        Minimum weight for including an edge (default=0.1). Edges below this
        weight will be excluded.

    Returns
    -------
    tuple
        A tuple of three lists: sources, destinations, and weights of edges.
    """
    import numpy as np

    # Get the matrix and corresponding indices
    matrix = pval_df.values
    n = matrix.shape[0]
    row_indices, col_indices = np.triu_indices(n, k=1)  # Upper triangle indices

    # Extract corresponding values
    weights = matrix[row_indices, col_indices]

    # Apply threshold to filter edges
    mask = weights > threshold
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    weights = weights[mask]

    # Convert indices back to labels
    sources = pval_df.index[row_indices].tolist()
    destinations = pval_df.columns[col_indices].tolist()
    weights = weights.tolist()

    return sources, destinations, weights

def build_node_index_map(node_labels):
    """
    Given a list of node labels (strings), build two dicts:
      label_to_id: str -> int
      id_to_label: int -> str
    """
    label_to_id = {}
    id_to_label = {}
    for i, lbl in enumerate(node_labels):
        label_to_id[lbl] = i
        id_to_label[i] = lbl
    return label_to_id, id_to_label

def prepare_edges_for_cpp(sources, destinations, weights):
    """
    Convert string-labeled edges to integer-labeled edges for the C++ random_walk function.
    Returns (row_int, col_int, weights_float, label_to_id, id_to_label).
    """
    # Collect all unique labels
    unique_nodes = set(sources).union(destinations)
    unique_nodes = list(unique_nodes)  # fix an order

    # Build maps
    label_to_id, id_to_label = build_node_index_map(unique_nodes)

    # Convert edges
    row_int = []
    col_int = []
    weights_float = []
    for s, d, w in zip(sources, destinations, weights):
        row_int.append(label_to_id[s])
        col_int.append(label_to_id[d])
        weights_float.append(float(w))

    return row_int, col_int, weights_float, label_to_id, id_to_label

def make_all_nodes_list(label_to_id):
    """
    Return a list of all node IDs (ints) to start random walks from.
    """
    # label_to_id is a dict {str -> int}
    sorted_pairs = sorted(label_to_id.items(), key=lambda x: x[1])
    # e.g. [(label, id), (label2, id2), ...] sorted by id
    # We just want the IDs in ascending order:
    start_nodes = [pair[1] for pair in sorted_pairs]
    return start_nodes



def run_biased_random_walk(row_int, col_int, weights_float, start_nodes,
                           walk_length=10, p=1.0, q=1.0, num_threads=1, walks_per_node=1):
    from ViruLink.random_walk import biased_random_walk
    """
    Wrap the C++ random_walk call. Returns a list of walks (each is a list of int node IDs).
    """
    
    walks = biased_random_walk.random_walk(
        row_int,
        col_int,
        start_nodes,
        weights_float,
        walk_length,
        p,
        q,
        num_threads,
        walks_per_node
    )
    return walks



def train_node2vec_embeddings(walks, vector_size=64, window=5, min_count=0, epochs=5):
    """
    Given a list of walks (each walk is a list of integer node-IDs),
    treat each node-ID as a "word" and each walk as a "sentence".
    Train a Word2Vec model to learn embeddings.

    Returns:
      model: a Gensim Word2Vec model
    """
    from gensim.models import Word2Vec
    
    # Convert each walk from list of int to list of str (because gensim expects tokens as strings)
    walks_str = [[str(node_id) for node_id in walk] for walk in walks]
    
    model = Word2Vec(
        sentences=walks_str,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,         # use skip-gram
        workers=4,    # number of threads for training
        epochs=epochs
    )
    return model

def get_embeddings(model, id_to_label):
    """
    Return a dict mapping { label_str : np.array(...) } for each node.
    """
    # Gensim keys are strings of the node ID
    emb_dict = {}
    for node_id, label_str in id_to_label.items():
        # node_id is int, label_str is the string label
        # But in the model, the key is str(node_id)
        key = str(node_id)
        if key in model.wv:
            emb_dict[label_str] = model.wv[key]
        else:
            # Possibly a node not encountered in any walk, or model didn't include it
            emb_dict[label_str] = None  # or a zero vector, etc.
    return emb_dict



def plot_tsne_embeddings(emb_dict):
    
    """
    Given a dict {label_str : embedding_vector}, 
    run t-SNE to reduce to 2D, then do an interactive Plotly scatter with hover = label.

    If some embeddings are None or have different sizes, skip them or handle accordingly.
    """
    from sklearn.manifold import TSNE
    import plotly.express as px
    labels = []
    vectors = []
    for lbl, vec in emb_dict.items():
        if vec is not None:
            labels.append(lbl)
            vectors.append(vec)

    vectors = np.array(vectors)
    # Run t-SNE
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=15)
    coords_2d = tsne.fit_transform(vectors)  # shape (num_nodes, 2)

    df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "label": labels
    })
    fig = px.scatter(
        df, x="x", y="y", hover_name="label",
        title="Node2Vec Embeddings (t-SNE)"
    )
    fig.show()


def logging_header(message: str, *args, width: int = 70):
    formatted = message % args if args else message
    centered = f" {formatted.upper()} ".center(width, "=")
    logging.info("=" * width)
    logging.info(centered)
    logging.info("=" * width)