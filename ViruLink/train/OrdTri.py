import torch
import torch.nn as nn

class OrdTri(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        # embeddings: a, b, a−b  → 3·dim
        # edge feats: ANI/HYP six-vector
        # probs for all three edges: 3 · k
        in_dim = dim * 3 + 6 + 3 * k
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128,  64),    nn.ReLU(),
            nn.Linear(64, k)                    # cumulative-logit logits
        )

    def forward(self, xa, xb, edge_feats, probs_vec):
        x = torch.cat([xa, xb, xa - xb, edge_feats, probs_vec], dim=-1)
        return self.net(x)
