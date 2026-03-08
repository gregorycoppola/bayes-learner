"""
Transformer for Bayesian posterior prediction.

No constraints on architecture — just needs to learn to map
factor graph encodings to exact posterior beliefs.

Input:  [batch, 3, 8]  — encoded factor graph (3 nodes, 8 features)
Output: [batch, 3]     — predicted P(x=1) for each node

Architecture: standard transformer encoder + per-node readout.
No positional encoding tricks, no construction hints.
Let gradient descent find the solution.
"""
import torch
import torch.nn as nn

D_MODEL = 8


class BPTransformer(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # Project input from 8 dims to d_model
        self.input_proj = nn.Linear(D_MODEL, d_model)
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Per-node readout → scalar belief
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 3, 8]
        returns: [batch, 3] beliefs in [0,1]
        """
        x = self.input_proj(x)           # [batch, 3, d_model]
        x = self.encoder(x)              # [batch, 3, d_model]
        out = self.readout(x).squeeze(-1)  # [batch, 3]
        return torch.sigmoid(out)