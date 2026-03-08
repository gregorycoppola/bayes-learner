"""
Two-head transformer matching the construction in Attention.lean.

d_model = 8
Head 0: Wq0/Wk0 project dim 1, Wv0 routes dim 0 → dim 4
Head 1: Wq1/Wk1 project dim 2, Wv1 routes dim 0 → dim 5
FFN:    reads dims 4,5, predicts updated belief

No LayerNorm — it would destroy the index values in dims 1 and 2
that the attention heads use for routing.

Constructed target weights (from Attention.lean):
  Wq0[1][1] = 1.0, all else 0  (projectDim(1))
  Wk0[1][1] = 1.0, all else 0  (projectDim(1))
  Wv0[4][0] = 1.0, all else 0  (crossProject(0→4))
  Wq1[2][2] = 1.0, all else 0  (projectDim(2))
  Wk1[2][2] = 1.0, all else 0  (projectDim(2))
  Wv1[5][0] = 1.0, all else 0  (crossProject(0→5))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8


# The constructed weight matrices from Attention.lean
def constructed_Wq0() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[1, 1] = 1.0
    return W

def constructed_Wk0() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[1, 1] = 1.0
    return W

def constructed_Wv0() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[4, 0] = 1.0
    return W

def constructed_Wq1() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[2, 2] = 1.0
    return W

def constructed_Wk1() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[2, 2] = 1.0
    return W

def constructed_Wv1() -> torch.Tensor:
    W = torch.zeros(D_MODEL, D_MODEL)
    W[5, 0] = 1.0
    return W

CONSTRUCTED = {
    "Wq0": (constructed_Wq0, (1, 1)),  # (constructor, expected argmax (row,col))
    "Wk0": (constructed_Wk0, (1, 1)),
    "Wv0": (constructed_Wv0, (4, 0)),
    "Wq1": (constructed_Wq1, (2, 2)),
    "Wk1": (constructed_Wk1, (2, 2)),
    "Wv1": (constructed_Wv1, (5, 0)),
}


class BPTransformer(nn.Module):
    """
    Two-head transformer for BP inference.
    Input:  [batch, n, 8]
    Output: [batch, n] beliefs in [0,1]
    """
    def __init__(self):
        super().__init__()
        # Separate Q/K/V per head — no fused projection
        # nn.Linear(in, out): weight shape [out, in]
        # so W[i][j] = weight from input dim j to output dim i
        # matches Lean convention: Wv[d][i] = 1 means read i, write d
        self.Wq0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wq1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        # FFN reads full 8-dim embedding after attention, predicts belief
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def attention_head(self, x, Wq, Wk, Wv):
        """
        Single attention head with softmax routing.
        x: [batch, n, d]
        returns: [batch, n, d] attended values (no residual here)
        """
        Q = Wq(x)  # [batch, n, d]
        K = Wk(x)  # [batch, n, d]
        V = Wv(x)  # [batch, n, d]
        scores = torch.bmm(Q, K.transpose(1, 2))  # [batch, n, n]
        attn   = F.softmax(scores, dim=-1)         # [batch, n, n]
        return torch.bmm(attn, V)                  # [batch, n, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Head 0: gather neighbor 0's belief into dim 4
        x = x + self.attention_head(x, self.Wq0, self.Wk0, self.Wv0)
        # Head 1: gather neighbor 1's belief into dim 5
        x = x + self.attention_head(x, self.Wq1, self.Wk1, self.Wv1)
        # FFN: read gathered beliefs, predict updated belief
        out = self.ffn(x).squeeze(-1)  # [batch, n]
        return torch.sigmoid(out)