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