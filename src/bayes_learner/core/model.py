"""
Two-head transformer for BP inference.

Key insight: dot-product attention can implement neighbor lookup when
Q·K peaks at the target neighbor. With normalized indices in [0,1]:
  score(j→k) = (nb_idx_j / (n-1)) * (own_idx_k / (n-1))
This is maximized when own_idx_k == nb_idx_j AND both are large.
It does NOT implement exact equality matching.

Temperature scaling amplifies score gaps, pushing softmax toward hardmax.
The Lean proof uses λ→∞ (hardmax limit). We use a large fixed temperature.

Constructed weights:
  Wq0[1,1]=1: query reads dim1 (neighbor0 index, normalized)
  Wk0[1,6]=1: key   reads dim6 (own index, normalized)  → score = nb0*own
  Wv0[4,0]=1: value reads dim0 (belief) → writes dim4 (scratch0)
  Wq1[2,2]=1: query reads dim2 (neighbor1 index, normalized)
  Wk1[2,6]=1: key   reads dim6 (own index, normalized)
  Wv1[5,0]=1: value reads dim0 (belief) → writes dim5 (scratch1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8
EPS = 1e-6


def logit(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p.clamp(EPS, 1-EPS) / (1 - p.clamp(EPS, 1-EPS)))


def constructed_Wq0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 1] = 1.0; return W

def constructed_Wk0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 6] = 1.0; return W

def constructed_Wv0():
    W = torch.zeros(D_MODEL, D_MODEL); W[4, 0] = 1.0; return W

def constructed_Wq1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 2] = 1.0; return W

def constructed_Wk1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 6] = 1.0; return W

def constructed_Wv1():
    W = torch.zeros(D_MODEL, D_MODEL); W[5, 0] = 1.0; return W

CONSTRUCTED = {
    "Wq0": (constructed_Wq0, (1, 1)),
    "Wk0": (constructed_Wk0, (1, 6)),
    "Wv0": (constructed_Wv0, (4, 0)),
    "Wq1": (constructed_Wq1, (2, 2)),
    "Wk1": (constructed_Wk1, (2, 6)),
    "Wv1": (constructed_Wv1, (5, 0)),
}

CONSTRUCTORS = {k: v[0] for k, v in CONSTRUCTED.items()}


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
        inp = torch.stack([b, m0, m1], dim=-1)
        return torch.sigmoid(self.net(inp).squeeze(-1))


class BPTransformer(nn.Module):
    def __init__(self, init: str = "constructed", noise: float = 0.01,
                 ffn_mode: str = "learned", temperature: float = 50.0):
        super().__init__()
        self.init_mode   = init
        self.ffn_mode    = ffn_mode
        self.temperature = temperature
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
        print(f"[MODEL] Initialized from Attention.lean construction "
              f"(noise={noise}, temperature={self.temperature})")

    def attention_head(self, x, Wq, Wk, Wv):
        Q = Wq(x)
        K = Wk(x)
        V = Wv(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.temperature
        attn   = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_head(x, self.Wq0, self.Wk0, self.Wv0)
        x = x + self.attention_head(x, self.Wq1, self.Wk1, self.Wv1)
        return self.ffn(x)