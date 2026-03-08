"""
Two-head transformer matching the construction in Attention.lean.

d_model = 8
Head 0: Wq0/Wk0 project dim 1, Wv0 routes dim 0 → dim 4
Head 1: Wq1/Wk1 project dim 2, Wv1 routes dim 0 → dim 5
FFN:    reads dims 4,5, predicts updated belief

No LayerNorm — it destroys the index values in dims 1 and 2.

Two init modes:
  random:      standard kaiming init (baseline — expected to fail)
  constructed: initialize from Attention.lean weights + small noise
               tests whether gradient descent stays near the construction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8


def constructed_Wq0():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[1, 1] = 1.0
    return W

def constructed_Wk0():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[1, 1] = 1.0
    return W

def constructed_Wv0():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[4, 0] = 1.0
    return W

def constructed_Wq1():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[2, 2] = 1.0
    return W

def constructed_Wk1():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[2, 2] = 1.0
    return W

def constructed_Wv1():
    W = torch.zeros(D_MODEL, D_MODEL)
    W[5, 0] = 1.0
    return W

CONSTRUCTED = {
    "Wq0": (constructed_Wq0, (1, 1)),
    "Wk0": (constructed_Wk0, (1, 1)),
    "Wv0": (constructed_Wv0, (4, 0)),
    "Wq1": (constructed_Wq1, (2, 2)),
    "Wk1": (constructed_Wk1, (2, 2)),
    "Wv1": (constructed_Wv1, (5, 0)),
}

CONSTRUCTORS = {
    "Wq0": constructed_Wq0,
    "Wk0": constructed_Wk0,
    "Wv0": constructed_Wv0,
    "Wq1": constructed_Wq1,
    "Wk1": constructed_Wk1,
    "Wv1": constructed_Wv1,
}


class BPTransformer(nn.Module):
    def __init__(self, init: str = "constructed", noise: float = 0.01):
        """
        init: "constructed" — start from Attention.lean weights + noise
              "random"      — standard kaiming init (expected to fail)
        noise: std of gaussian noise added to constructed weights
        """
        super().__init__()
        self.init_mode = init
        self.Wq0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wq1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        if init == "constructed":
            self._init_from_construction(noise)

    def _init_from_construction(self, noise: float):
        for name, constructor in CONSTRUCTORS.items():
            W = constructor()
            if noise > 0:
                W = W + torch.randn_like(W) * noise
            getattr(self, name).weight.data.copy_(W)
        print(f"[MODEL] Initialized from Attention.lean construction (noise={noise})")

    def attention_head(self, x, Wq, Wk, Wv):
        Q = Wq(x)
        K = Wk(x)
        V = Wv(x)
        scores = torch.bmm(Q, K.transpose(1, 2))
        attn   = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_head(x, self.Wq0, self.Wk0, self.Wv0)
        x = x + self.attention_head(x, self.Wq1, self.Wk1, self.Wv1)
        out = self.ffn(x).squeeze(-1)
        return torch.sigmoid(out)