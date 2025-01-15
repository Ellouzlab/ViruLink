import os
import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba


def visualize_taxonomy_network(taxa_df, family_labels, output_dir, filename='taxonomy_network.png', num_nodes=500):
    """
    Visualizes the taxonomy network using NetworkX with an option to subsample nodes.

    Args:
        taxa_df (pd.DataFrame): Taxonomy similarity matrix (adjacency matrix).
        family_labels (pd.Series): Family labels for each genome.
        output_dir (str): Directory to save the visualization.
        filename (str): Filename for the visualization.
        num_nodes (int): Number of nodes to subsample for visualization.

    Returns:
        None
    """
    try:
        logging.info("Starting taxonomy network visualization.")

        # Filter out nodes with unknown or NaN family labels
        valid_nodes = family_labels[~family_labels.isin(['unknown', np.nan])].dropna().index
        taxa_df = taxa_df.loc[valid_nodes, valid_nodes]
        family_labels = family_labels.loc[valid_nodes]
        logging.info(f"Excluded nodes with 'unknown' or NaN family labels. Remaining nodes: {len(valid_nodes)}")

        # Subsample nodes if the number of nodes exceeds the limit
        if len(taxa_df) > num_nodes:
            sampled_nodes = np.random.choice(taxa_df.index, size=num_nodes, replace=False)
            taxa_df = taxa_df.loc[sampled_nodes, sampled_nodes]
            family_labels = family_labels.loc[sampled_nodes]
            logging.info(f"Subsampled to {num_nodes} nodes for visualization.")

        # Remove self-loops by setting diagonal elements to 0
        np.fill_diagonal(taxa_df.values, 0)
        logging.info("Removed self-loops from the adjacency matrix.")

        # Create a NetworkX graph from the adjacency matrix
        G = nx.Graph()

        # Add edges with weights greater than 1
        for i, row in taxa_df.iterrows():
            for j, weight in row.items():
                if weight > 1:  # Exclude edges with weights 0 or 1
                    G.add_edge(i, j, weight=weight)

        # Add family labels as node attributes
        nx.set_node_attributes(G, family_labels.to_dict(), name='family')

        # Map family labels to colors using matplotlib colormap
        unique_families = family_labels.unique()
        num_families = len(unique_families)
        cmap = get_cmap('tab20')  # Use the 'tab20' colormap
        family_to_color = {family: cmap(i / num_families) for i, family in enumerate(unique_families)}

        # Assign node colors based on families
        node_colors = [family_to_color[family_labels[node]] for node in G.nodes]

        # Extract edge weights for visualization
        edge_weights = nx.get_edge_attributes(G, 'weight')
        edge_weights = np.array(list(edge_weights.values()))

        # Normalize edge weights for visualization purposes
        if len(edge_weights) > 0:
            min_weight, max_weight = np.min(edge_weights), np.max(edge_weights)
            normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight + 1e-8)
            edge_widths = normalized_weights * 2 + 0.1  # Thinner edges
        else:
            edge_widths = []

        # Draw the graph
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)  # Using spring layout for clear visualization

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=50,
            alpha=0.8
        )

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.3,
            width=edge_widths
        )

        # Add legend for families
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=to_rgba(color), markersize=8, label=family)
            for family, color in family_to_color.items()
        ]
        plt.legend(handles=legend_elements, title="Families", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add title and save the plot
        plt.title("Taxonomy Network")
        plt.axis('off')
        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        logging.info(f"Taxonomy network visualization saved at {plot_path}")
    except Exception as e:
        logging.error(f"Error in visualize_taxonomy_network: {e}")
        raise
