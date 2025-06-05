import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --- SwiGLU Activation ---
class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act, gate = x.chunk(2, dim=-1)
        return act * F.silu(gate)

# --- Stage 1: Independent Edge Feature Extractor & Predictor (Same as before GNN) ---
class EdgeFeatureNet(nn.Module):
    def __init__(self, comb_dim: int, k_classes: int, act: str = "relu", 
                 output_feature_dim: int = 32): # comb_dim is the input node embedding dim
        super().__init__()
        self.k_classes = k_classes
        self.act_name = act.lower()
        self.output_feature_dim = output_feature_dim

        in_dim_edge_processor = 3 * comb_dim + 2 
        hidden_dim_edge1 = 64 
        hidden_dim_edge2 = output_feature_dim

        if self.act_name == "relu":
            self.feature_extractor = nn.Sequential(
                nn.Linear(in_dim_edge_processor, hidden_dim_edge1), nn.ReLU(), nn.BatchNorm1d(hidden_dim_edge1),
                nn.Linear(hidden_dim_edge1, hidden_dim_edge2), nn.ReLU(), nn.BatchNorm1d(hidden_dim_edge2)
            )
            self.logit_predictor = nn.Linear(hidden_dim_edge2, k_classes)
        elif self.act_name == "swiglu":
            self.feature_extractor = nn.Sequential(
                nn.Linear(in_dim_edge_processor, hidden_dim_edge1 * 2), SwiGLU(), nn.BatchNorm1d(hidden_dim_edge1),
                nn.Linear(hidden_dim_edge1, hidden_dim_edge2 * 2), SwiGLU(), nn.BatchNorm1d(hidden_dim_edge2)
            )
            self.logit_predictor = nn.Linear(hidden_dim_edge2, k_classes)
        else: raise ValueError("act must be 'relu' or 'swiglu'")

    def forward(self, xa_emb: torch.Tensor, xb_emb: torch.Tensor, 
                edge_specific_raw_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_diff = xa_emb - xb_emb
        edge_input = torch.cat([xa_emb, xb_emb, x_diff, edge_specific_raw_feats], dim=-1)
        learned_edge_representation = self.feature_extractor(edge_input)
        initial_edge_logits = self.logit_predictor(learned_edge_representation)
        return initial_edge_logits, learned_edge_representation

# --- Stage 2: Triangle Refinement Network with Self-Attention ---
class AttentionTriangleRefinerNet(nn.Module):
    def __init__(self, k_classes: int, node_emb_dim: int, edge_feature_dim: int, # node_emb_dim is COMB_DIM
                 attn_heads: int, attn_layers: int, attn_dropout: float, act: str = "relu"):
        super().__init__()
        self.k_classes = k_classes
        self.act_name = act.lower()
        self.edge_feature_dim = edge_feature_dim # Dimension of features from EdgeFeatureNet

        # Transformer Encoder Layer for self-attention over edge features
        # The input to TransformerEncoderLayer should be (seq_len, batch_size, feature_dim)
        # We have 3 edges, so seq_len = 3.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=edge_feature_dim, 
            nhead=attn_heads, 
            dim_feedforward=edge_feature_dim * 2, # Typical feedforward expansion
            dropout=attn_dropout,
            activation=F.relu if act == "relu" else F.gelu, # GELU is common in transformers
            batch_first=False # Expects (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        # MLP to process combined features after attention
        # Input to MLP:
        # 3 * contextualized_edge_features (from transformer) -> 3 * edge_feature_dim
        # All 3 original vertex embeddings -> 3 * node_emb_dim
        # Triangle 6-vector raw features -> 6
        # prev_pass_refined_probs_all_edges (concatenated adjacent probs) -> 3 * k_classes
        
        mlp_in_dim = (3 * edge_feature_dim) + \
                     (3 * node_emb_dim) + \
                     6 + \
                     (3 * k_classes)

        hidden_dim_mlp1 = 256 
        hidden_dim_mlp2 = 128

        if self.act_name == "relu":
            self.final_mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, hidden_dim_mlp1), nn.ReLU(), nn.BatchNorm1d(hidden_dim_mlp1), nn.Dropout(0.15),
                nn.Linear(hidden_dim_mlp1, hidden_dim_mlp2), nn.ReLU(), nn.BatchNorm1d(hidden_dim_mlp2), nn.Dropout(0.15),
                nn.Linear(hidden_dim_mlp2, 3 * k_classes) 
            )
        elif self.act_name == "swiglu": # Using SwiGLU for the final MLP as well
             self.final_mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, hidden_dim_mlp1 * 2), SwiGLU(), nn.BatchNorm1d(hidden_dim_mlp1), nn.Dropout(0.15),
                nn.Linear(hidden_dim_mlp1, hidden_dim_mlp2 * 2), SwiGLU(), nn.BatchNorm1d(hidden_dim_mlp2), nn.Dropout(0.15),
                nn.Linear(hidden_dim_mlp2, 3 * k_classes)
            )
        else: raise ValueError("act must be 'relu' or 'swiglu'")


    def forward(self,
                qr1_learned_edge_feats: torch.Tensor, # from EdgeFeatureNet [B, edge_feature_dim]
                qr2_learned_edge_feats: torch.Tensor, 
                r1r2_learned_edge_feats: torch.Tensor,
                eq_emb: torch.Tensor,       # Original node embedding [B, node_emb_dim]
                ea_emb: torch.Tensor,      
                eh_emb: torch.Tensor,      
                triangle_all_edge_raw_feats: torch.Tensor, # [B, 6]
                prev_pass_refined_adj_probs: torch.Tensor  # [B, 3*k_classes]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = qr1_learned_edge_feats.size(0)

        # Prepare input for Transformer Encoder: (seq_len, batch_size, feature_dim)
        # seq_len = 3 (for the three edges)
        edge_features_stacked = torch.stack([
            qr1_learned_edge_feats, qr2_learned_edge_feats, r1r2_learned_edge_feats
        ], dim=0) # Shape: [3, B, edge_feature_dim]

        # Self-attention over edge features
        # No mask needed as all edges can attend to all others.
        contextualized_edge_features_stacked = self.transformer_encoder(edge_features_stacked) 
        # Output shape: [3, B, edge_feature_dim]

        # Flatten contextualized edge features for MLP input: [B, 3 * edge_feature_dim]
        contextualized_edge_features_flat = contextualized_edge_features_stacked.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Combine with other features for the final MLP
        combined_input_for_mlp = torch.cat([
            contextualized_edge_features_flat,
            eq_emb, ea_emb, eh_emb, # Original node embeddings
            triangle_all_edge_raw_feats,
            prev_pass_refined_adj_probs
        ], dim=-1)
        
        all_refined_logits = self.final_mlp(combined_input_for_mlp) # Shape [B, 3*k_classes]
        
        la_final, lh_final, lr_final = torch.split(all_refined_logits, self.k_classes, dim=1)
        return la_final, lh_final, lr_final

# --- Main Two-Stage Model with Attention-based Refiner ---
class OrdTriTwoStageAttn(nn.Module):
    def __init__(self, node_emb_dim: int, # This is COMB_DIM
                 k_classes: int, act: str = "relu", 
                 edge_feature_dim: int = 32, # Output of EdgeFeatureNet, input to Attention
                 attn_heads: int = 4, attn_layers: int = 1, attn_dropout: float = 0.1):
        super().__init__()
        # EdgeFeatureNet uses node_emb_dim (e.g. COMB_DIM) as input for node representations
        self.edge_predictor = EdgeFeatureNet(node_emb_dim, k_classes, act, edge_feature_dim)
        
        # AttentionTriangleRefinerNet uses node_emb_dim for original vertex embeddings
        # and edge_feature_dim for features from EdgeFeatureNet
        self.triangle_refiner = AttentionTriangleRefinerNet(
            k_classes, node_emb_dim, edge_feature_dim, 
            attn_heads, attn_layers, attn_dropout, act
        )

    def forward(self,
                eq_emb: torch.Tensor, ea_emb: torch.Tensor, eh_emb: torch.Tensor, # Original embeddings
                triangle_all_edge_raw_feats: torch.Tensor,
                prev_pass_refined_adj_probs: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, # Final logits
                          torch.Tensor, torch.Tensor, torch.Tensor  # Initial logits
                         ]:
        # Stage 1: Independent edge predictions
        qr1_specific_feats = triangle_all_edge_raw_feats[:, 0:2]
        qr1_initial_logits, qr1_learned_feats = self.edge_predictor(eq_emb, ea_emb, qr1_specific_feats)
        
        qr2_specific_feats = triangle_all_edge_raw_feats[:, 2:4]
        qr2_initial_logits, qr2_learned_feats = self.edge_predictor(eq_emb, eh_emb, qr2_specific_feats)

        r1r2_specific_feats = triangle_all_edge_raw_feats[:, 4:6]
        r1r2_initial_logits, r1r2_learned_feats = self.edge_predictor(ea_emb, eh_emb, r1r2_specific_feats)

        # Stage 2: Triangle Refinement with Attention
        la_final, lh_final, lr_final = self.triangle_refiner(
            qr1_learned_feats, qr2_learned_feats, r1r2_learned_feats,
            eq_emb, ea_emb, eh_emb, # Pass original embeddings to refiner
            triangle_all_edge_raw_feats,
            prev_pass_refined_adj_probs
        )
        
        return la_final, lh_final, lr_final, \
               qr1_initial_logits, qr2_initial_logits, r1r2_initial_logits