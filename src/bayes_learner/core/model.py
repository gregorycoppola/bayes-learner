"""
Two-head transformer matching the construction in Attention.lean.

Attention weights (learned, match construction):
  Head 0: Wq0/Wk0 project dim 1 → routes neighbor 0's belief to dim 4
  Head 1: Wq1/Wk1 project dim 2 → routes neighbor 1's belief to dim 5

FFN (BP update in log-odds space):
  BP update: new_belief = sigmoid(logit(b) + logit(msg0) + logit(msg1))
  where b = dim 0, msg0 = dim 4, msg1 = dim 5
  This is the correct inductive bias for belief propagation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 8
EPS = 1e-6


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(EPS, 1 - EPS)
    return torch.log(p / (1 - p))


def constructed_Wq0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 1] = 1.0; return W

def constructed_Wk0():
    W = torch.zeros(D_MODEL, D_MODEL); W[1, 1] = 1.0; return W

def constructed_Wv0():
    W = torch.zeros(D_MODEL, D_MODEL); W[4, 0] = 1.0; return W

def constructed_Wq1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 2] = 1.0; return W

def constructed_Wk1():
    W = torch.zeros(D_MODEL, D_MODEL); W[2, 2] = 1.0; return W

def constructed_Wv1():
    W = torch.zeros(D_MODEL, D_MODEL); W[5, 0] = 1.0; return W

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


class BPUpdateFFN(nn.Module):
    """
    FFN that computes the BP belief update.

    Two modes:
      'learned'     — standard MLP, learns the update from data
      'constructed' — exact BP update: sigmoid(logit(b) + logit(m0) + logit(m1))
                      no learned parameters, directly implements the formula

    The constructed version verifies the attention is working correctly.
    The learned version tests whether gradient descent finds the BP update.
    """
    def __init__(self, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            self.net = nn.Sequential(
                nn.Linear(3, 32),   # inputs: dim0 (belief), dim4 (msg0), dim5 (msg1)
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n, 8]
        returns: [batch, n] updated beliefs
        """
        b   = x[:, :, 0]  # dim 0: current belief
        m0  = x[:, :, 4]  # dim 4: neighbor 0's belief (gathered by head 0)
        m1  = x[:, :, 5]  # dim 5: neighbor 1's belief (gathered by head 1)

        if self.mode == "constructed":
            # Exact BP update in log-odds space
            return torch.sigmoid(logit(b) + logit(m0) + logit(m1))
        else:
            # Learned: feed [b, m0, m1] to MLP
            inp = torch.stack([b, m0, m1], dim=-1)  # [batch, n, 3]
            return torch.sigmoid(self.net(inp).squeeze(-1))


class BPTransformer(nn.Module):
    def __init__(self, init: str = "constructed", noise: float = 0.01,
                 ffn_mode: str = "learned"):
        """
        init:     "constructed" | "random" — attention weight initialization
        ffn_mode: "learned"     — MLP learns BP update from data
                  "constructed" — exact BP formula, no parameters (oracle)
        """
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
        # Attention: gather neighbor beliefs into dims 4 and 5
        x = x + self.attention_head(x, self.Wq0, self.Wk0, self.Wv0)
        x = x + self.attention_head(x, self.Wq1, self.Wk1, self.Wv1)
        # FFN: compute updated belief from dims 0, 4, 5
        return self.ffn(x)