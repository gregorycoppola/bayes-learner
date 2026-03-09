"""
Chain of 3 variables: v0 --- f1 --- v1 --- f2 --- v2

Requires 2 rounds of BP for exact posteriors.

Key fix: v1 gets BOTH factor tables explicitly in its encoding.
No attention lookup needed for f2's table — all information local.

Encoding (d=16):
  [0]  own_belief (init 0.5)
  [1]  neighbor_0_index / (n-1)
  [2]  neighbor_1_index / (n-1)   (0 if no second neighbor)
  [3]  node_type (0=variable, 1=factor)
  [4]  own_index / (n-1)
  [5]  ft_left[0,0]
  [6]  ft_left[0,1]
  [7]  ft_left[1,0]
  [8]  ft_left[1,1]
  [9]  ft_right[0,0]   (0 if no right factor)
  [10] ft_right[0,1]
  [11] ft_right[1,0]
  [12] ft_right[1,1]
  [13-15] reserved (zeros)

v1 is the only node with both ft_left and ft_right populated.
n=5 nodes, indices 0..4, normalized by 4.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import Tuple, List

D_MODEL = 16
N_NODES = 5


def _exact_bp_2rounds(ft1: List[float], ft2: List[float]) -> Tuple[float, float, float]:
    b = [0.5, 0.5, 0.5, 0.5, 0.5]

    def update_belief(m0, m1):
        num = m0 * m1
        den = num + (1 - m0) * (1 - m1)
        return num / den if den > 0 else 0.5

    def factor_message(ft, b_in):
        m1 = ft[1] * (1 - b_in) + ft[3] * b_in
        m0 = ft[0] * (1 - b_in) + ft[2] * b_in
        z = m0 + m1
        return m1 / z if z > 0 else 0.5

    for _ in range(2):
        msg_f1_to_v0 = factor_message(ft1, b[2])
        msg_f1_to_v1 = factor_message(ft1, b[0])
        msg_f2_to_v1 = factor_message(ft2, b[4])
        msg_f2_to_v2 = factor_message(ft2, b[2])
        b[0] = update_belief(msg_f1_to_v0, 0.5)
        b[2] = update_belief(msg_f1_to_v1, msg_f2_to_v1)
        b[4] = update_belief(msg_f2_to_v2, 0.5)

    return b[0], b[2], b[4]


def _encode_ft(emb: torch.Tensor, row: int, start: int, ft: List[float]):
    emb[row, start]     = ft[0]
    emb[row, start + 1] = ft[1]
    emb[row, start + 2] = ft[2]
    emb[row, start + 3] = ft[3]


@dataclass
class TwoNeighborChain:
    ft1: List[float]   # [f(0,0), f(0,1), f(1,0), f(1,1)]
    ft2: List[float]

    def encode(self) -> torch.Tensor:
        ft1, ft2 = self.ft1, self.ft2
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: v0 — one neighbor (f1)
        emb[0, 0] = 0.5
        emb[0, 1] = 1.0 / 4.0   # neighbor_0 = f1 (index 1)
        emb[0, 2] = 0.0          # no second neighbor
        emb[0, 3] = 0.0          # variable
        emb[0, 4] = 0.0 / 4.0   # own index
        _encode_ft(emb, 0, 5, ft1)
        # dims 9-15 = zeros

        # Token 1: f1 — factor, neighbors v0(0) and v1(2)
        emb[1, 0] = 0.5
        emb[1, 1] = 0.0 / 4.0   # neighbor_0 = v0
        emb[1, 2] = 2.0 / 4.0   # neighbor_1 = v1
        emb[1, 3] = 1.0          # factor
        emb[1, 4] = 1.0 / 4.0   # own index
        _encode_ft(emb, 1, 5, ft1)
        # dims 9-15 = zeros

        # Token 2: v1 — two neighbors (f1 and f2), gets BOTH tables
        emb[2, 0] = 0.5
        emb[2, 1] = 1.0 / 4.0   # neighbor_0 = f1 (index 1)
        emb[2, 2] = 3.0 / 4.0   # neighbor_1 = f2 (index 3)
        emb[2, 3] = 0.0          # variable
        emb[2, 4] = 2.0 / 4.0   # own index
        _encode_ft(emb, 2, 5, ft1)   # left factor table
        _encode_ft(emb, 2, 9, ft2)   # right factor table — KEY FIX

        # Token 3: f2 — factor, neighbors v1(2) and v2(4)
        emb[3, 0] = 0.5
        emb[3, 1] = 2.0 / 4.0   # neighbor_0 = v1
        emb[3, 2] = 4.0 / 4.0   # neighbor_1 = v2
        emb[3, 3] = 1.0          # factor
        emb[3, 4] = 3.0 / 4.0   # own index
        _encode_ft(emb, 3, 5, ft2)
        # dims 9-15 = zeros

        # Token 4: v2 — one neighbor (f2)
        emb[4, 0] = 0.5
        emb[4, 1] = 3.0 / 4.0   # neighbor_0 = f2 (index 3)
        emb[4, 2] = 0.0          # no second neighbor
        emb[4, 3] = 0.0          # variable
        emb[4, 4] = 4.0 / 4.0   # own index
        _encode_ft(emb, 4, 5, ft2)
        # dims 9-15 = zeros

        return emb

    def exact_posteriors(self) -> Tuple[float, float, float]:
        return _exact_bp_2rounds(self.ft1, self.ft2)


def make_graph() -> TwoNeighborChain:
    ft1 = [random.uniform(0.05, 1.0) for _ in range(4)]
    ft2 = [random.uniform(0.05, 1.0) for _ in range(4)]
    return TwoNeighborChain(ft1=ft1, ft2=ft2)


def make_dataset(n_graphs: int, log_every: int = 2000):
    t0 = time.time()
    graphs = []
    for i in range(n_graphs):
        graphs.append(make_graph())
        if (i + 1) % log_every == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n_graphs - i - 1) / rate
            print(f"[DATA]   {i+1}/{n_graphs} "
                  f"({rate:.0f}/s, ~{remaining:.1f}s remaining)")

    X = torch.stack([g.encode() for g in graphs])

    Y = torch.zeros(n_graphs, N_NODES)
    for i, g in enumerate(graphs):
        v0, v1, v2 = g.exact_posteriors()
        Y[i, 0] = v0
        Y[i, 1] = 0.5
        Y[i, 2] = v1
        Y[i, 3] = 0.5
        Y[i, 4] = v2

    var_mask = torch.zeros(n_graphs, N_NODES, dtype=torch.bool)
    var_mask[:, 0] = True
    var_mask[:, 2] = True
    var_mask[:, 4] = True

    return X, Y, var_mask