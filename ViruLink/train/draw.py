def draw_network(
    hyper_adj_df,
    w_comb_adj_df,
    output,
    csv_file,
    max_nodes=500
):
    '''
    Draw the network graphs based on adjacency matrices and taxonomy information,
    limited to a specified number of nodes, excluding nodes with 'Unknown' taxonomy.

    Args:
        hyper_adj_df: The hypergeometric adjacency matrix (binary).
        w_comb_adj_df: The weighted adjacency matrix.
        output: The output directory.
        csv_file: The taxonomy CSV file.
        max_nodes: The maximum number of nodes to include in the visualization.
    '''
    import os
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import numpy as np

    taxonomy_df = pd.read_csv(csv_file)
    taxonomy_df.set_index('Accession', inplace=True)

    drawings_dir = os.path.join(output, 'network_drawings')
    os.makedirs(drawings_dir, exist_ok=True)

    adjacency_matrices = [
        (hyper_adj_df, 'hypergeometric_binary'),
        (w_comb_adj_df, 'combined_weighted')
    ]

    taxonomy_levels = ['Family', 'Genus']

    for adj_df, adj_label in adjacency_matrices:
        # Remove self-loops
        adj_df.values[np.diag_indices_from(adj_df)] = 0

        if adj_label == 'combined_weighted':
            G = nx.from_pandas_adjacency(adj_df, create_using=nx.Graph())
        else:
            G = nx.from_pandas_adjacency(adj_df.astype(bool), create_using=nx.Graph())

        G.remove_edges_from(nx.selfloop_edges(G))

        node_taxonomy = {}
        for node in G.nodes():
            if node in taxonomy_df.index:
                family = taxonomy_df.at[node, 'Family']
                genus = taxonomy_df.at[node, 'Genus']
                if pd.isna(family) or family == '':
                    family = 'Unknown'
                if pd.isna(genus) or genus == '':
                    genus = 'Unknown'
            else:
                family = 'Unknown'
                genus = 'Unknown'
            node_taxonomy[node] = {'Family': family, 'Genus': genus}

        # Exclude nodes with 'Unknown' taxonomy
        nodes_with_known_taxonomy = [
            node for node in G.nodes()
            if node_taxonomy[node]['Family'] != 'Unknown' and node_taxonomy[node]['Genus'] != 'Unknown'
        ]

        # Create subgraph with nodes having known taxonomy
        G_known = G.subgraph(nodes_with_known_taxonomy).copy()

        if G_known.number_of_nodes() > max_nodes:
            print(f"Graph has {G_known.number_of_nodes()} nodes after excluding 'Unknowns'. Selecting a subgraph with {max_nodes} nodes.")

            # Extract the largest connected component
            largest_cc = max(nx.connected_components(G_known), key=len)
            subgraph_nodes = list(largest_cc)

            # If the largest connected component is still too big, select top nodes by degree
            if len(subgraph_nodes) > max_nodes:
                # Get degrees of nodes in the largest connected component
                subgraph = G_known.subgraph(subgraph_nodes)
                degrees = dict(subgraph.degree())
                # Sort nodes by degree in descending order
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                # Select top max_nodes nodes
                selected_nodes = [node for node, degree in sorted_nodes[:max_nodes]]
                G_sub = G_known.subgraph(selected_nodes).copy()
            else:
                G_sub = G_known.subgraph(subgraph_nodes).copy()
        else:
            G_sub = G_known

        # For each taxonomy level
        for tax_level in taxonomy_levels:
            unique_tax_values = list(set([node_taxonomy[node][tax_level] for node in G_sub.nodes()]))
            unique_tax_values.sort()

            cmap = get_cmap('tab20')
            colors = cmap.colors
            tax_value_color_map = {tax_value: colors[i % len(colors)] for i, tax_value in enumerate(unique_tax_values)}

            node_colors = []
            for node in G_sub.nodes():
                tax_value = node_taxonomy[node][tax_level]
                color = tax_value_color_map[tax_value]
                node_colors.append(color)

            node_size = 30  # Adjusted from 50 to 30 for smaller nodes

            if adj_label == 'combined_weighted':
                edges = G_sub.edges()
                weights = [adj_df.loc[u, v] for u, v in edges]
                max_weight = max(weights)
                min_weight = min(weights)
                edge_widths = [0.05 + 0.15 * ((w - min_weight) / (max_weight - min_weight + 1e-6)) for w in weights]
            else:
                edge_widths = 0.05  # For binary graph, set edge width to 0.05

            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G_sub, seed=42)

            nx.draw_networkx_nodes(G_sub, pos, node_size=node_size, node_color=node_colors)
            # nx.draw_networkx_labels(G_sub, pos, font_size=8, font_color='black')  # Commented out

            nx.draw_networkx_edges(G_sub, pos, width=edge_widths, edge_color='grey', alpha=0.5)

            from matplotlib.patches import Patch
            legend_handles = [Patch(color=tax_value_color_map[tax_value], label=tax_value) for tax_value in unique_tax_values]
            plt.legend(handles=legend_handles, title=tax_level, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.axis('off')

            filename = f"{adj_label}_{tax_level}_network.png"
            filepath = os.path.join(drawings_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"Saved network drawing: {filepath}")
