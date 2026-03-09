"""
Chain with clean two-neighbor encoding for v1.

    v0 --- f1 --- v1 --- f2 --- v2

Same graph as exp003 but with a corrected encoding that explicitly
gives v1 both neighbor indices without overloading any slot.

Key insight: ft[1,1] is dropped from the encoding. It is recoverable
as (1 - ft[0,0] - ft[0,1] - ft[1,0]) only if the table is normalized,
which it isn't here (unnormalized positive reals). So we instead drop
ft[1,1] and accept the information loss — the transformer can still
compute exact posteriors because the marginals only require the ratio
of row sums, not the absolute values.

Actually: exact posteriors DO use ft[1,1]. So dropping it is lossy.
Instead we drop ft[0,0] — the transformer can infer relative weights
from the other three entries well enough, and attention index matching
is more important than having all 4 factor entries.

Encoding (d=8):
  [0] own_belief (init 0.5)
  [1] neighbor_0_index / (n-1)    ← clean index slot
  [2] neighbor_1_index / (n-1)    ← clean index slot (0 if no second neighbor)
  [3] node_type (0=variable, 1=factor)
  [4] ft[0,1]   (was ft[0,0] — dropped to free dim for neighbor)
  [5] ft[1,0]
  [6] own_index / (n-1)
  [7] ft[1,1]

ft[0,0] is dropped. The transformer sees 3 of 4 factor table entries.
This is the minimal change that preserves clean index matching while
fitting in d=8.

n=5 nodes, indices 0..4, normalized by 4.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import Tuple, List

D_MODEL = 8
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


@dataclass
class TwoNeighborChain:
    ft1: List[float]   # [f(0,0), f(0,1), f(1,0), f(1,1)]
    ft2: List[float]

    def encode(self) -> torch.Tensor:
        ft1, ft2 = self.ft1, self.ft2
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: v0 — neighbor_0=f1(1), no second neighbor
        emb[0, 0] = 0.5
        emb[0, 1] = 1.0 / 4.0   # neighbor_0 = f1
        emb[0, 2] = 0.0          # no second neighbor
        emb[0, 3] = 0.0          # variable
        emb[0, 4] = ft1[1]       # ft[0,1]
        emb[0, 5] = ft1[2]       # ft[1,0]
        emb[0, 6] = 0.0 / 4.0   # own index
        emb[0, 7] = ft1[3]       # ft[1,1]

        # Token 1: f1 — neighbors: v0(0), v1(2)
        emb[1, 0] = 0.5
        emb[1, 1] = 0.0 / 4.0   # neighbor_0 = v0
        emb[1, 2] = 2.0 / 4.0   # neighbor_1 = v1
        emb[1, 3] = 1.0          # factor
        emb[1, 4] = ft1[1]
        emb[1, 5] = ft1[2]
        emb[1, 6] = 1.0 / 4.0   # own index
        emb[1, 7] = ft1[3]

        # Token 2: v1 — neighbor_0=f1(1), neighbor_1=f2(3) ← KEY NODE
        emb[2, 0] = 0.5
        emb[2, 1] = 1.0 / 4.0   # neighbor_0 = f1
        emb[2, 2] = 3.0 / 4.0   # neighbor_1 = f2  ← explicit
        emb[2, 3] = 0.0          # variable
        emb[2, 4] = ft1[1]
        emb[2, 5] = ft1[2]
        emb[2, 6] = 2.0 / 4.0   # own index
        emb[2, 7] = ft1[3]

        # Token 3: f2 — neighbors: v1(2), v2(4)
        emb[3, 0] = 0.5
        emb[3, 1] = 2.0 / 4.0   # neighbor_0 = v1
        emb[3, 2] = 4.0 / 4.0   # neighbor_1 = v2
        emb[3, 3] = 1.0          # factor
        emb[3, 4] = ft2[1]
        emb[3, 5] = ft2[2]
        emb[3, 6] = 3.0 / 4.0   # own index
        emb[3, 7] = ft2[3]

        # Token 4: v2 — neighbor_0=f2(3), no second neighbor
        emb[4, 0] = 0.5
        emb[4, 1] = 3.0 / 4.0   # neighbor_0 = f2
        emb[4, 2] = 0.0          # no second neighbor
        emb[4, 3] = 0.0          # variable
        emb[4, 4] = ft2[1]
        emb[4, 5] = ft2[2]
        emb[4, 6] = 4.0 / 4.0   # own index
        emb[4, 7] = ft2[3]

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