import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    x is split into (act, gate) halves; output = act * SiLU(gate).
    If the input has shape (..., 2·d) the output has shape (..., d).
    """
    def forward(self, x):
        act, gate = x.chunk(2, dim=-1)
        return act * F.silu(gate)


class OrdTri(nn.Module):
    """
    Ordinal-triangle edge head.

    Parameters
    ----------
    dim  : int   – embedding dimension per genome
    k    : int   – number of ordinal ranks
    act  : str   – 'relu' (default) or 'swiglu'
    """
    def __init__(self, dim: int, k: int, act: str = "relu") -> None:
        super().__init__()
        self.k = k
        self.act_name = act.lower()

        # embeddings: a, b, a−b  → 3·dim
        # edge feats: ANI/HYP six-vector
        # probs for all three edges: 3 · k
        in_dim = dim * 3 + 6 + 3 * k

        if self.act_name == "relu":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128,  64),    nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, k)                 # cumulative-logit logits
            )
        elif self.act_name == "swiglu":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128 * 2), SwiGLU(), nn.Dropout(0.1),    # → 128
                nn.Linear(128,      64 * 2), SwiGLU(), nn.Dropout(0.1),     # → 64
                nn.Linear(64, k)                          # logits
            )
        else:
            raise ValueError("act must be 'relu' or 'swiglu'")

    # ------------------------------------------------------------------
    def forward(
        self,
        xa: torch.Tensor,
        xb: torch.Tensor,
        edge_feats: torch.Tensor,
        probs_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        xa, xb     : genome embeddings               – shape [B, dim]
        edge_feats : six ANI/HYP scalars             – shape [B, 6]
        probs_vec  : concatenated previous probs     – shape [B, 3·k]
        """
        x = torch.cat([xa, xb, xa - xb, edge_feats, probs_vec], dim=-1)
        return self.net(x)
