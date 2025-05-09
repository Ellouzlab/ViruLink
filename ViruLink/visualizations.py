# ViruLink/visualizations.py
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def WeightvRelation(rel_df: pd.DataFrame,
                    wts_df: pd.DataFrame,
                    x_nodes: int = 1000,
                    seed: int | None = None) -> None:
    """
    Scatter‑plot lower / upper relationship codes vs. weight
    for ≈ x_nodes sampled nodes.  Adds linear trend lines plus
    equation & Pearson‑r for each series.
    """

    rng = random.Random(seed)

    # ── 1. edge‑first sampling until ≥ x_nodes nodes ────────────────
    sampled_edges, keep_nodes = [], set()
    idx_pool = list(range(len(wts_df)))
    rng.shuffle(idx_pool)

    for idx in idx_pool:
        row = wts_df.iloc[idx]
        keep_nodes.update((row["source"], row["target"]))
        sampled_edges.append(idx)
        if len(keep_nodes) >= x_nodes:
            break

    rel = rel_df[rel_df["source"].isin(keep_nodes) &
                 rel_df["target"].isin(keep_nodes)]
    wts = wts_df.iloc[sampled_edges]

    merged = rel.merge(wts, on=["source", "target"]).dropna(subset=["weight"])
    if merged.empty:
        raise RuntimeError("edge sampling produced zero overlap")

    # ── 2. scatter plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["upper"], merged["weight"],
               s=10, alpha=0.6, label="upper", color="C0")
    ax.scatter(merged["lower"], merged["weight"],
               s=10, alpha=0.6, label="lower", color="C1")

    # ── 3. trend lines + stats ─────────────────────────────────────
    stat_lines = []
    for col, color, label in [("upper", "C0", "upper"),
                              ("lower", "C1", "lower")]:
        x = merged[col].astype(float).to_numpy()
        y = merged["weight"].astype(float).to_numpy()

        if np.unique(x).size > 1:
            m, b = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]

            # plot line
            x_fit = np.array([x.min(), x.max()])
            ax.plot(x_fit, m * x_fit + b,
                    color=color, linewidth=2, alpha=0.4,
                    label=f"{label} trend")

            stat_lines.append(f"{label}: y = {m:.3g}x + {b:.3g},  r = {r:.3g}")

    # ── 4. labels, legend, annotation ──────────────────────────────
    ax.set_xlabel("Relationship code (higher = closer)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Weight vs. relationship bounds  (sample ≈ {x_nodes} nodes)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    # Put the stats in the bottom‑right corner
    '''
    if stat_lines:
        ax.text(0.98, 0.02, "\n".join(stat_lines),
                ha="left", va="bottom", transform=ax.transAxes,
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", alpha=0.8))'''

    fig.tight_layout()
    plt.show()

    # also print to stdout/log
    for line in stat_lines:
        print(line)






def draw_taxonomic_subgraphs(df, x, metadata_path='/home/sulman/.cache/ViruLink/Caudoviricetes/Caudoviricetes.csv', seed=None):
    """
    Draws 5 random subgraphs (same nodes), colored by different taxonomic levels.
    Treats missing or blank metadata as 'Unknown', always colored gray.
    Saves each subgraph as a separate PNG file.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    import pandas as pd
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Clean node names
    df['source'] = df['source'].str.split('.').str[0]
    df['target'] = df['target'].str.split('.').str[0]
    df = df[df["source"] != df["target"]]

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    print(f"Metadata loaded from {metadata_path}")
    print(f"Columns available for coloring: {metadata_df.columns.tolist()}")

    # Build full graph
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.Graph())
    all_nodes = list(G.nodes)
    if len(all_nodes) < x:
        raise ValueError(f"Graph only has {len(all_nodes)} nodes, but x={x} requested.")

    # Sample fixed node set
    random.seed(seed)
    sampled_nodes = random.sample(all_nodes, x)
    subG = G.subgraph(sampled_nodes).copy()
    subG.remove_edges_from(nx.selfloop_edges(subG))
    pos = nx.spring_layout(subG, seed=seed, k=0.4)

    # Prepare node metadata
    node_df = pd.DataFrame({'Accession': list(subG.nodes)})
    merged = node_df.merge(metadata_df, on='Accession', how='left')

    levels = ["Family", "Genus", "Subfamily", "Species", "Order"]

    for level in levels:
        # Normalize missing and blank values to 'Unknown'
        taxa = merged[level].fillna("").replace("", "Unknown")

        # Extract unique known taxa
        known_taxa = sorted(t for t in taxa.unique() if t != "Unknown")
        color_map = {"Unknown": "gray"}

        # Assign distinct colors to known taxa
        cmap = cm.get_cmap('tab20', len(known_taxa))
        for i, taxon in enumerate(known_taxa):
            color_map[taxon] = cmap(i)

        # Map node colors
        node_colors = [color_map[t] for t in taxa]

        # Get edge weights
        weights = [subG[u][v]['weight'] for u, v in subG.edges()]
        max_weight = max(weights) if weights else 1.0  # prevent div by zero
        scaled_weights = [0.2 * w / max_weight for w in weights]

        # Plot graph
        plt.figure(figsize=(5, 5))
        nx.draw_networkx_nodes(subG, pos, node_size=3, node_color=node_colors)
        nx.draw_networkx_edges(subG, pos, edge_color='gray', width=scaled_weights)

        # Legend
        for taxon, color in color_map.items():
            plt.scatter([], [], c=[color], label=taxon, s=30)

        # Save output
        plt.title(f"Random Subgraph ({x} nodes) colored by {level}")
        plt.axis('off')
        plt.tight_layout()
        base_path = metadata_path.split('.csv')[0]
        save_path = f"{base_path}_subgraph_{level}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    return subG

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