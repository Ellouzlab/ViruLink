import torch, os, logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler, label_binarize
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=2, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False, edge_dim=2, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
    
    def forward(self, x):
        return self.MLP(x)


def prepare_training_tensors(hyper_adj_df, weighted_adj_df, taxa_df):
    try:
        logging.info("Preparing training tensors.")

        shared_indices = list(set(hyper_adj_df.index) & set(weighted_adj_df.index) & set(taxa_df.index))
        hyper_adj_df = hyper_adj_df.loc[shared_indices, shared_indices].fillna(0) 
        weighted_adj_df = weighted_adj_df.loc[shared_indices, shared_indices].fillna(0) 
        taxa_df = taxa_df.loc[shared_indices, shared_indices].fillna(-1) 

        logging.info(f"Shared indices across matrices: {len(shared_indices)}")

        genome_to_idx = {genome: idx for idx, genome in enumerate(shared_indices)}

        hyper_values = hyper_adj_df.stack().reset_index()
        weighted_values = weighted_adj_df.stack().reset_index()

        hyper_values.columns = ['source', 'target', 'hyper']
        weighted_values.columns = ['source', 'target', 'weighted']

        edge_features = pd.merge(hyper_values, weighted_values, on=['source', 'target'])
        edge_features = edge_features[(edge_features['hyper'] != 0) & (edge_features['weighted'] != 0)]

        edge_features['source'] = edge_features['source'].map(genome_to_idx)
        edge_features['target'] = edge_features['target'].map(genome_to_idx)
        
        edge_features_tensor = torch.tensor(edge_features[['source', 'target', 'hyper', 'weighted']].values, dtype=torch.float32)
        logging.info("Filtered edge features tensor created with two-dimensional attributes.")

        link_values = taxa_df.stack().reset_index()
        link_values.columns = ['source', 'target', 'link_type']
        link_values['source'] = link_values['source'].map(genome_to_idx)
        link_values['target'] = link_values['target'].map(genome_to_idx)
        link_types_tensor = torch.tensor(link_values[['source', 'target', 'link_type']].values, dtype=torch.float32)
        logging.info("Link types tensor created.")

        return edge_features_tensor, link_types_tensor, genome_to_idx

    except Exception as e:
        logging.error(f"Error in prepare_training_tensors: {e}")
        raise


def visualize_forward_pass(encoder, presence_absence_tensor, edge_features_tensor, family_labels, output_dir, filename='forward_tsne.png'):
    try:
        logging.info("Starting forward pass visualization.")

        device = next(encoder.parameters()).device 
        presence_absence_tensor = presence_absence_tensor.to(device)
        
        edge_index = edge_features_tensor[:, :2].T.long().to(device)
        edge_attr = edge_features_tensor[:, 2:].to(device)

        encoder.eval()
        with torch.no_grad():
            node_embeddings = encoder(presence_absence_tensor, edge_index, edge_attr)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        tsne_results = tsne.fit_transform(node_embeddings.cpu().numpy())

        tsne_df = pd.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Family': family_labels.values
        })

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
        plt.title('t-SNE Visualization of Node Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

        logging.info(f"t-SNE visualization saved at {plot_path}")

    except Exception as e:
        logging.error(f"Error in visualize_forward_pass: {e}")
        raise


def evaluate(encoder, decoder, links, edge_index, edge_attr, presence_absence_tensor, device, batch_size, name, return_preds=False):
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0
    true_labels = []
    pred_labels = []
    pred_probs = []

    logging.info(f"Evaluating {name}...")
    for i in tqdm(range(0, links.size(0), batch_size), desc=f"{name} Batches"):
        batch = links[i:i + batch_size].to(device)
        src_nodes, dst_nodes, labels = batch[:, 0].long(), batch[:, 1].long(), batch[:, 2].long() - 1

        with autocast(device_type=device):
            node_embeddings = encoder(presence_absence_tensor, edge_index, edge_attr)
            src_emb = node_embeddings[src_nodes]
            dst_emb = node_embeddings[dst_nodes]
            link_emb = torch.cat([src_emb, dst_emb], dim=1)
            link_pred = decoder(link_emb)
            probs = F.softmax(link_pred, dim=1)

        preds = link_pred.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if return_preds:
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            pred_probs.extend(probs.detach().cpu().numpy())

    accuracy = correct / total

    if return_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, labels=[0,1,2], average=None, zero_division=0
        )
        logging.info(f"{name} Class-Specific Metrics:")
        for cls_idx, (prec, rec_, f1_score) in enumerate(zip(precision, recall, f1), start=1):
            logging.info(f"Class {cls_idx}: Precision={prec:.4f}, Recall={rec_:.4f}, F1-Score={f1_score:.4f}")
        return accuracy, true_labels, pred_labels, pred_probs, precision, recall, f1

    return accuracy


def propagate_labels(encoder, decoder, presence_absence_tensor, edge_index, edge_attr, labels_series, label_type="family", device='cpu'):
    """
    Propagate labels for unknown nodes using the nearest known node.
    Accuracy = correct / total_non_none_predictions
    Recall   = correct / total_unknown_nodes

    For family: propagate if pred_class in [2,3].
    For genus: propagate only if pred_class == 3.

    Additionally, prints incorrect predictions showing true and predicted labels.
    """

    # Filter out nodes with NaN labels
    valid_indices = labels_series.dropna().index
    valid_nodes = np.array([i for i in range(len(labels_series)) if labels_series.index[i] in valid_indices])

    # Shuffle and split these valid nodes
    np.random.shuffle(valid_nodes)
    test_size = int(0.1 * len(valid_nodes))
    unknown_nodes = valid_nodes[:test_size]
    known_nodes = valid_nodes[test_size:]

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        node_embeddings = encoder(presence_absence_tensor.to(device), edge_index.to(device), edge_attr.to(device))

    normalized_embeddings = F.normalize(node_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
    distance_matrix = 1.0 - similarity_matrix

    predicted_labels = {}
    incorrect_predictions = []  # List to store incorrect predictions

    for node_idx in unknown_nodes:
        known_distances = distance_matrix[node_idx, known_nodes]
        closest_idx_rel = torch.argmin(known_distances).item()
        closest_known_node = known_nodes[closest_idx_rel]

        with torch.no_grad():
            pair_emb = torch.cat([node_embeddings[node_idx], node_embeddings[closest_known_node]], dim=0).unsqueeze(0)
            link_pred = decoder(pair_emb)
            pred_class = torch.argmax(link_pred, dim=1).item() + 1

        # Logic depends on label_type
        if label_type == "family":
            # Propagate if class is 2 or 3
            if pred_class in [2, 3]:
                predicted_labels[node_idx] = labels_series.iloc[closest_known_node]
            else:
                predicted_labels[node_idx] = None
        elif label_type == "genus":
            # Propagate only if class is 3
            if pred_class == 3:
                predicted_labels[node_idx] = labels_series.iloc[closest_known_node]
            else:
                predicted_labels[node_idx] = None
        else:
            # Default: treat as family logic
            if pred_class in [2, 3]:
                predicted_labels[node_idx] = labels_series.iloc[closest_known_node]
            else:
                predicted_labels[node_idx] = None

    # Get true labels and reindex them with unknown_nodes
    true_labels = labels_series.iloc[unknown_nodes].copy()
    true_labels.index = unknown_nodes

    # Count how many were correct and collect incorrect predictions
    correct = 0
    for n in unknown_nodes:
        if predicted_labels[n] is not None:
            if predicted_labels[n] == true_labels.loc[n]:
                correct += 1
            else:
                # Collect incorrect predictions
                incorrect_predictions.append({
                    'Node Index': n,
                    'True Label': true_labels.loc[n],
                    'Predicted Label': predicted_labels[n]
                })
                print(f"Incorrect Prediction - Node {n}: True Label = {true_labels.loc[n]}, Predicted Label = {predicted_labels[n]}")
        else:
            # No prediction was made
            pass

    # Count how many had non-None predictions
    non_none_predictions = [n for n in unknown_nodes if predicted_labels[n] is not None]
    total_non_none = len(non_none_predictions)

    # Total unknown nodes
    total_unknown_nodes = len(unknown_nodes)

    # Accuracy = correct / total_non_none_predictions
    accuracy = correct / total_non_none if total_non_none > 0 else 0.0

    # Recall = correct / total_unknown_nodes
    recall = correct / total_unknown_nodes if total_unknown_nodes > 0 else 0.0

    return accuracy, recall




def plot_comparison_bar(family_acc, family_rec, genus_acc, genus_rec, output_dir):
    metrics = ['Accuracy', 'Recall']
    family_values = [family_acc, family_rec]
    genus_values = [genus_acc, genus_rec]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8,6))
    plt.bar(x - width/2, family_values, width, label='Family')
    plt.bar(x + width/2, genus_values, width, label='Genus')

    plt.xticks(x, metrics)
    plt.ylabel('Score')
    plt.title('Label Propagation Comparison (Family vs Genus)')
    plt.ylim(0,1)  # since accuracy and recall are between 0 and 1
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'family_genus_propagation_comparison.png'), dpi=300)
    plt.close()


def plot_training_loss(train_losses, output_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300)
    plt.close()


def plot_validation_accuracy(val_accuracies, output_dir):
    plt.figure()
    plt.plot(val_accuracies, label='Validation Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.ylim(0,1)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'validation_accuracy.png'), dpi=300)
    plt.close()


def plot_multiclass_roc_pr(true_labels, pred_probs, output_dir, classes=[1,2,3]):
    y_true = np.array(true_labels)
    Y = label_binarize(y_true, classes=[0,1,2])  
    Y_score = np.array(pred_probs)  

    # ROC Curves
    plt.figure(figsize=(10,8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], Y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title('Multi-class ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'multi_class_roc.png'), dpi=300)
    plt.close()

    # PR Curves
    plt.figure(figsize=(10,8))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(Y[:, i], Y_score[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {cls} (AUC = {pr_auc:.2f})')
    plt.title('Multi-class Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, 'multi_class_pr.png'), dpi=300)
    plt.close()


def train_model(hyper_adj_df, weighted_adj_df, taxa_df, family_labels, genus_labels, presence_absence_df, output_dir, edge_batch_size=1024):
    try:
        logging.info("Starting memory-optimized training setup with random node features.")

        edge_features_tensor, link_types_tensor, genome_to_idx = prepare_training_tensors(
            hyper_adj_df=hyper_adj_df,
            weighted_adj_df=weighted_adj_df,
            taxa_df=taxa_df
        )

        # Filter out unknown links and class 4
        valid_links = (link_types_tensor[:, 2] != 0) & (link_types_tensor[:, 2] != 4)
        link_types_tensor = link_types_tensor[valid_links]

        # Classes: 1,2,3
        class_1_mask = link_types_tensor[:, 2] == 1.0
        class_2_mask = link_types_tensor[:, 2] == 2.0
        class_3_mask = link_types_tensor[:, 2] == 3.0

        class_1_indices = torch.where(class_1_mask)[0]
        class_2_indices = torch.where(class_2_mask)[0]
        class_3_indices = torch.where(class_3_mask)[0]

        # Downsample classes to handle class imbalance
        sampled_class_1 = class_1_indices[torch.randperm(len(class_1_indices))[: len(class_1_indices) // 20]]
        sampled_class_2 = class_2_indices[torch.randperm(len(class_2_indices))[: len(class_2_indices) // 2]]
        sampled_class_3 = class_3_indices

        sampled_indices = torch.cat([sampled_class_1, sampled_class_2, sampled_class_3])
        link_types_tensor = link_types_tensor[sampled_indices]

        link_classes, link_counts = torch.unique(link_types_tensor[:, 2], return_counts=True)
        logging.info(f"Number of links for each class after sampling: {dict(zip(link_classes.cpu().numpy(), link_counts.cpu().numpy()))}")

        aligned_family_labels = family_labels.loc[genome_to_idx.keys()]
        aligned_genus_labels = genus_labels.loc[genome_to_idx.keys()]

        # Instead of using presence_absence_df, we initialize random features.
        num_nodes = len(genome_to_idx)
        input_dim = 1024  # You can choose a suitable dimension for random features
        hidden_dim = 64
        output_dim = 32
        dropout = 0.2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Randomly initialize node features
        presence_absence_tensor = torch.randn((num_nodes, input_dim), dtype=torch.float32)

        encoder = GATEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout).to(device)
        decoder = Decoder(input_dim=output_dim * 2, hidden_dim=hidden_dim, output_dim=3, dropout=dropout).to(device)

        edge_index = edge_features_tensor[:, :2].T.long().to(device)
        edge_attr = edge_features_tensor[:, 2:].to(device)
        presence_absence_tensor = presence_absence_tensor.to(device)

        num_links = link_types_tensor.size(0)
        indices = torch.randperm(num_links)
        train_split = int(0.8 * num_links)
        val_split = int(0.9 * num_links)

        train_links = link_types_tensor[indices[:train_split]]
        val_links = link_types_tensor[indices[train_split:val_split]]
        test_links = link_types_tensor[indices[val_split:]]

        logging.info("Data split into train, validation, and test sets.")

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        epochs = 10
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            encoder.train()
            decoder.train()

            total_loss = 0
            logging.info(f"Epoch {epoch + 1}/{epochs} - Training...")
            for i in tqdm(range(0, train_links.size(0), edge_batch_size), desc=f"Epoch {epoch + 1} Batches"):
                optimizer.zero_grad()

                batch = train_links[i:i + edge_batch_size].to(device)
                src_nodes = batch[:, 0].long()
                dst_nodes = batch[:, 1].long()
                labels = batch[:, 2].long() - 1

                with autocast(device_type=device):
                    node_embeddings = encoder(presence_absence_tensor, edge_index, edge_attr)
                    src_emb = node_embeddings[src_nodes]
                    dst_emb = node_embeddings[dst_nodes]
                    link_emb = torch.cat([src_emb, dst_emb], dim=1)
                    link_pred = decoder(link_emb)

                    loss = criterion(link_pred, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
            train_losses.append(total_loss)

            val_accuracy = evaluate(encoder, decoder, val_links, edge_index, edge_attr, presence_absence_tensor, device, edge_batch_size, "Validation")
            logging.info(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")
            val_accuracies.append(val_accuracy)

        # Plot training curves in separate figures
        plot_training_loss(train_losses, output_dir)
        plot_validation_accuracy(val_accuracies, output_dir)

        # Evaluate on test
        test_accuracy, true_labels, pred_labels, pred_probs, precision, recall, f1 = evaluate(
            encoder, decoder, test_links, edge_index, edge_attr, presence_absence_tensor, device, edge_batch_size, "Test", return_preds=True
        )

        logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")
        logging.info("Final Test Class-Specific Metrics:")
        for cls_idx, (prec, rec_, f1_score) in enumerate(zip(precision, recall, f1), start=1):
            logging.info(f"Class {cls_idx}: Precision={prec:.4f}, Recall={rec_:.4f}, F1-Score={f1_score:.4f}")

        cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2], normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 1", "Class 2", "Class 3"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        logging.info("Confusion matrix saved.")

        plot_multiclass_roc_pr(true_labels, pred_probs, output_dir, classes=[1,2,3])

        # Visualize embeddings using t-SNE
        visualize_forward_pass(encoder, presence_absence_tensor, edge_features_tensor, aligned_family_labels, output_dir, filename='forward_tsne.png')

        # Run label propagation tests for both family and genus
        family_acc, family_rec = propagate_labels(encoder, decoder, presence_absence_tensor, edge_index, edge_attr, aligned_family_labels, label_type="family", device=device)
        genus_acc, genus_rec = propagate_labels(encoder, decoder, presence_absence_tensor, edge_index, edge_attr, aligned_genus_labels, label_type="genus", device=device)

        logging.info(f"Family label propagation accuracy: {family_acc:.4f}, recall: {family_rec:.4f}")
        logging.info(f"Genus label propagation accuracy: {genus_acc:.4f}, recall: {genus_rec:.4f}")

        # Compare family and genus propagation in a figure
        plot_comparison_bar(family_acc, family_rec, genus_acc, genus_rec, output_dir)

    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise
