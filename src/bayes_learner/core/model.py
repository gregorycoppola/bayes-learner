"""
Transformer matching the construction in Attention.lean.

Architecture:
  d_model = 8
  2 attention heads
  Head 0: Q/K on dim 1 (neighbor 0 index), V: dim 0 → dim 4
  Head 1: Q/K on dim 2 (neighbor 1 index), V: dim 0 → dim 5
  FFN: reads dims 4,5 (gathered neighbor beliefs), writes updated belief to dim 0

This matches the explicit weight construction proven in transformer-bp-lean.
"""
import torch
import torch.nn as nn

D_MODEL = 8
K = 2


class BPTransformer(nn.Module):
    """
    Single-round BP transformer.
    Input:  [batch, n, 8] encoded factor graph state
    Output: [batch, n]    predicted beliefs after one BP round
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = K):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # Standard multi-head attention — let gradient descent find the weights
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            bias=False,
        )
        # FFN: 2-layer, reads all 8 dims, outputs 8 dims
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Final readout: extract belief from dim 0
        self.readout = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n, 8]
        returns: [batch, n] belief predictions in [0,1]
        """
        # Attention + residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        # Readout: one scalar per node
        out = self.readout(x).squeeze(-1)  # [batch, n]
        return torch.sigmoid(out)