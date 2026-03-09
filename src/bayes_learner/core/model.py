"""
Transformer for Bayesian posterior prediction.

Input:  [batch, n_nodes, d_in]  — encoded factor graph
Output: [batch, n_nodes]        — predicted P(x=1) for each node

Architecture: standard transformer encoder + per-node readout.
No positional encoding tricks, no construction hints.
Let gradient descent find the solution.
"""
import torch
import torch.nn as nn


class BPTransformer(nn.Module):
    def __init__(self, d_in: int = 8, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        return torch.sigmoid(self.readout(x).squeeze(-1))