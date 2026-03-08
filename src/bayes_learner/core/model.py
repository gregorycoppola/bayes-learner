"""
Two-head transformer matching the construction in Attention.lean.

CORRECTED ENCODING:
  dim 1 = neighbor 0 index → Wq0 queries on this (projectDim(1))
  dim 2 = neighbor 1 index → Wq1 queries on this (projectDim(2))
  dim 6 = own index        → Wk0, Wk1 key on this (projectDim(6))

Score for head 0: Q·K = emb[1] * emb[6] = (nb0_index) * (own_index)
Peaks when own_index == nb0_index, i.e. when we're looking at neighbor 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8
EPS = 1e-6


def logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p.clamp(EPS, 1 - EPS) / (1 - p.clamp(EPS, 1 - EPS)))


def constructed_Wq0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 1] = 1.0; return W  # query on dim 1

def constructed_Wk0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 6] = 1.0; return W  # key: dim1 output ← dim6 input

def constructed_Wv0():
    W = torch.zeros(D_MODEL, D_MODEL); W[4, 0] = 1.0; return W  # belief → scratch 0

def constructed_Wq1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 2] = 1.0; return W  # query on dim 2

def constructed_Wk1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 6] = 1.0; return W  # key: dim2 output ← dim6 input

def constructed_Wv1():
    W = torch.zeros(D_MODEL, D_MODEL); W[5, 0] = 1.0; return W  # belief → scratch 1

CONSTRUCTED = {
    "Wq0": (constructed_Wq0, (1, 1)),
    "Wk0": (constructed_Wk0, (1, 6)),
    "Wv0": (constructed_Wv0, (4, 0)),
    "Wq1": (constructed_Wq1, (2, 2)),
    "Wk1": (constructed_Wk1, (2, 6)),
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


class BPUpdateFFN(nn.Module):
    def __init__(self, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.net = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b  = x[:, :, 0]
        m0 = x[:, :, 4]
        m1 = x[:, :, 5]
        if self.mode == "constructed":
            return torch.sigmoid(logit(b) + logit(m0) + logit(m1))
        else:
            inp = torch.stack([b, m0, m1], dim=-1)
            return torch.sigmoid(self.net(inp).squeeze(-1))


class BPTransformer(nn.Module):
    def __init__(self, init: str = "constructed", noise: float = 0.01,
                 ffn_mode: str = "learned"):
        super().__init__()
        self.init_mode = init
        self.ffn_mode  = ffn_mode
        self.Wq0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv0 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wq1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv1 = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ffn = BPUpdateFFN(mode=ffn_mode)
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
        return self.ffn(x)