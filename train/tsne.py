# tsne.py

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch.optim import SparseAdam  # Changed from Adam to SparseAdam

def plot_tsne_presence_absence(presence_absence_df, family_labels, output_dir, filename='tsne_presence_absence.png'):
    """
    Performs t-SNE on gene presence-absence data and plots the scatter plot.

    Args:
        presence_absence_df (pd.DataFrame): Gene presence-absence matrix.
        family_labels (pd.Series): Family labels for each genome.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the t-SNE plot.

    Returns:
        None
    """
    try:
        logging.info("Starting t-SNE for Gene Presence-Absence Data.")

        # Ensure that presence_absence_df and family_labels have the same length
        if len(presence_absence_df) != len(family_labels):
            raise ValueError(f"Length mismatch: presence_absence_df has {len(presence_absence_df)} samples, "
                             f"but family_labels has {len(family_labels)} samples.")

        # Ensure that the indices are aligned
        if not presence_absence_df.index.equals(family_labels.index):
            logging.warning("Indices of presence_absence_df and family_labels do not match. Aligning them.")
            family_labels = family_labels.loc[presence_absence_df.index]
            if family_labels.isnull().any():
                raise ValueError("Some family labels are missing after alignment.")

        # Scaling the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(presence_absence_df.values)
        logging.info("Data scaling completed.")

        # Performing t-SNE
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)  # Changed 'n_iter' to 'max_iter'
        tsne_results = tsne.fit_transform(scaled_data)
        logging.info("t-SNE dimensionality reduction completed.")

        # Creating a DataFrame for plotting
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Family': family_labels.values
        })

        # Plotting
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=tsne_df,
            x='TSNE1',
            y='TSNE2',
            hue='Family',
            palette='tab20',
            s=50,
            alpha=0.7
        )
        plt.title('t-SNE of Gene Presence-Absence Data')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Saving the plot
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"t-SNE plot for Gene Presence-Absence Data saved at {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_tsne_presence_absence: {e}")
        raise

def perform_node2vec_and_plot_tsne(hyper_adj_df, family_labels, output_dir, filename='tsne_node2vec_bfs.png',
                                   embedding_dim=64, walk_length=30, context_size=10, walks_per_node=10,
                                   num_negative_samples=1, p=1.0, q=0.5, epochs=10, device='cuda'):
    """
    Trains a BFS-like Node2Vec model, generates embeddings, performs t-SNE, and plots the scatter plot.

    Args:
        hyper_adj_df (pd.DataFrame): Hypergeometric adjacency matrix.
        family_labels (pd.Series): Family labels for each genome.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the t-SNE plot.
        embedding_dim (int): Dimension of node embeddings.
        walk_length (int): Length of each random walk.
        context_size (int): Context size for optimization.
        walks_per_node (int): Number of walks per node.
        num_negative_samples (int): Number of negative samples.
        p (float): Return hyperparameter.
        q (float): Inout hyperparameter.
        epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        None
    """
    try:
        logging.info("Starting BFS Node2Vec Embedding Generation.")

        # Ensure adjacency matrix aligns with family labels
        logging.info("Filtering adjacency matrix to align with family labels.")
        shared_nodes = hyper_adj_df.index.intersection(family_labels.index)
        hyper_adj_df = hyper_adj_df.loc[shared_nodes, shared_nodes]
        family_labels = family_labels.loc[shared_nodes]
        logging.info(f"Filtered adjacency matrix to {len(shared_nodes)} nodes.")

        # Convert adjacency matrix to edge list
        logging.info("Converting adjacency matrix to edge list.")
        edge_list = hyper_adj_df.stack().reset_index()
        edge_list.columns = ['source', 'target', 'weight']
        edge_list = edge_list[edge_list['source'] != edge_list['target']]
        edge_list = edge_list[edge_list['weight'] > 0]  # Remove zero-weight edges
        logging.info(f"Number of edges after processing: {len(edge_list)}")

        # Mapping genome IDs to integer indices
        genomes = shared_nodes.tolist()
        genome_to_idx = {genome: idx for idx, genome in enumerate(genomes)}
        logging.info("Mapping genome IDs to integer indices.")

        edge_list['source'] = edge_list['source'].map(genome_to_idx)
        edge_list['target'] = edge_list['target'].map(genome_to_idx)

        # Ensure no NaNs after mapping
        if edge_list['source'].isnull().any() or edge_list['target'].isnull().any():
            raise ValueError("Some source or target genomes could not be mapped to indices.")

        # Convert to integer type
        edge_list['source'] = edge_list['source'].astype(int)
        edge_list['target'] = edge_list['target'].astype(int)

        # Create PyTorch Geometric Data object
        edge_index = torch.tensor(edge_list[['source', 'target']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_list['weight'].values, dtype=torch.float).unsqueeze(1)
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Initialize Node2Vec model
        node2vec = Node2Vec(
            edge_index=data.edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=True
        ).to(device)

        # Define optimizer
        optimizer = SparseAdam(list(node2vec.parameters()), lr=0.01)

        # Training loop
        logging.info("Starting Node2Vec training.")
        node2vec.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for pos_rw, neg_rw in node2vec.loader():
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Node2Vec Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}")

        # Get embeddings
        node_embeddings = node2vec.embedding.weight.detach().cpu().numpy()
        logging.info("Node2Vec training completed.")

        # Ensure that node_embeddings length matches family_labels
        if node_embeddings.shape[0] != len(family_labels):
            raise ValueError(f"Length mismatch: Node embeddings have {node_embeddings.shape[0]} samples, "
                             f"but family_labels has {len(family_labels)} samples.")

        # Performing t-SNE
        logging.info("Starting t-SNE for Node2Vec Embeddings.")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(node_embeddings)
        logging.info("t-SNE dimensionality reduction completed.")

        # Creating a DataFrame for plotting
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Family': family_labels.values
        })

        # Plotting
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=tsne_df,
            x='TSNE1',
            y='TSNE2',
            hue='Family',
            palette='tab20',
            s=50,
            alpha=0.7
        )
        plt.title('t-SNE of BFS Node2Vec Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Saving the plot
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"t-SNE plot for BFS Node2Vec Embeddings saved at {plot_path}")

    except Exception as e:
        logging.error(f"Error in perform_node2vec_and_plot_tsne: {e}")
        raise

def perform_bfs_node2vec_and_plot_tsne(weighted_adj_df, family_labels, output_dir, filename='tsne_node2vec_bfs_weighted.png',
                                       embedding_dim=64, walk_length=30, context_size=10, walks_per_node=10,
                                       num_negative_samples=1, p=1.0, q=0.5, epochs=10, device='cuda'):
    """
    Trains a BFS-like Node2Vec model using a weighted adjacency matrix, generates embeddings, performs t-SNE, and plots.

    Args:
        weighted_adj_df (pd.DataFrame): Weighted adjacency matrix.
        family_labels (pd.Series): Family labels for each genome.
        output_dir (str): Directory to save the plot.
        filename (str): Filename for the t-SNE plot.
        embedding_dim (int): Dimension of node embeddings.
        walk_length (int): Length of each random walk.
        context_size (int): Context size for optimization.
        walks_per_node (int): Number of walks per node.
        num_negative_samples (int): Number of negative samples.
        p (float): Return hyperparameter.
        q (float): Inout hyperparameter.
        epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        None
    """
    try:
        logging.info("Starting BFS Node2Vec Embedding Generation with Weighted Adjacency Matrix.")

        # Convert adjacency matrix to edge list
        logging.info("Converting adjacency matrix to edge list.")
        edge_list = weighted_adj_df.stack().reset_index()
        edge_list.columns = ['source', 'target', 'weight']
        edge_list = edge_list[edge_list['source'] != edge_list['target']]
        edge_list = edge_list[edge_list['weight'] > 0]  # Remove zero-weight edges
        logging.info(f"Number of edges after processing: {len(edge_list)}")

        # Mapping genome IDs to integer indices
        genomes = weighted_adj_df.index.tolist()
        genome_to_idx = {genome: idx for idx, genome in enumerate(genomes)}
        logging.info("Mapping genome IDs to integer indices.")

        edge_list['source'] = edge_list['source'].map(genome_to_idx)
        edge_list['target'] = edge_list['target'].map(genome_to_idx)

        # Ensure no NaNs after mapping
        if edge_list['source'].isnull().any() or edge_list['target'].isnull().any():
            raise ValueError("Some source or target genomes could not be mapped to indices.")

        # Convert to integer type
        edge_list['source'] = edge_list['source'].astype(int)
        edge_list['target'] = edge_list['target'].astype(int)

        # Create PyTorch Geometric Data object
        edge_index = torch.tensor(edge_list[['source', 'target']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_list['weight'].values, dtype=torch.float).unsqueeze(1)
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        # Initialize Node2Vec model
        node2vec = Node2Vec(
            edge_index=data.edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=True
        ).to(device)

        # Define optimizer
        optimizer = SparseAdam(list(node2vec.parameters()), lr=0.01)

        # Training loop
        logging.info("Starting Node2Vec training.")
        node2vec.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for pos_rw, neg_rw in node2vec.loader():
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(f"Node2Vec Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}")

        # Get embeddings
        node_embeddings = node2vec.embedding.weight.detach().cpu().numpy()
        logging.info("Node2Vec training completed.")

        # Perform t-SNE
        logging.info("Starting t-SNE for Node2Vec Embeddings.")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(node_embeddings)
        logging.info("t-SNE dimensionality reduction completed.")

        # Create a DataFrame for plotting
        tsne_df = pd.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Family': family_labels.values
        })

        # Plotting
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=tsne_df,
            x='TSNE1',
            y='TSNE2',
            hue='Family',
            palette='tab20',
            s=50,
            alpha=0.7
        )
        plt.title('t-SNE of BFS Node2Vec Embeddings (Weighted)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"t-SNE plot for BFS Node2Vec Embeddings (Weighted) saved at {plot_path}")

    except Exception as e:
        logging.error(f"Error in perform_bfs_node2vec_and_plot_tsne: {e}")
        raise