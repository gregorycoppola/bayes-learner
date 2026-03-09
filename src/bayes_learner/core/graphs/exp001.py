"""
Minimal factor graph: two variable nodes connected by one factor node.

    v0 --- f1 --- v2

This is the simplest non-trivial Bayesian network.
Exact BP converges in one round — perfect for a single transformer pass.

Encoding per node (d=8):
  [0] own_belief (init 0.5)
  [1] factor_table[0,0] = P(x0=0, x1=0)
  [2] factor_table[0,1] = P(x0=0, x1=1)
  [3] factor_table[1,0] = P(x0=1, x1=0)
  [4] factor_table[1,1] = P(x0=1, x1=1)
  [5] node_type (0=variable, 1=factor)
  [6] own_index / (n-1)
  [7] neighbor_index / (n-1)
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

D_MODEL = 8


@dataclass
class TwoVarGraph:
    factor_table: List[float]

    def encode(self) -> torch.Tensor:
        ft = self.factor_table
        emb = torch.zeros(3, D_MODEL)
        emb[0, 0] = 0.5
        emb[0, 1] = ft[0]
        emb[0, 2] = ft[1]
        emb[0, 3] = ft[2]
        emb[0, 4] = ft[3]
        emb[0, 5] = 0.0
        emb[0, 6] = 0.0
        emb[0, 7] = 0.5

        emb[1, 0] = 0.5
        emb[1, 1] = ft[0]
        emb[1, 2] = ft[1]
        emb[1, 3] = ft[2]
        emb[1, 4] = ft[3]
        emb[1, 5] = 1.0
        emb[1, 6] = 0.5
        emb[1, 7] = 0.0

        emb[2, 0] = 0.5
        emb[2, 1] = ft[0]
        emb[2, 2] = ft[1]
        emb[2, 3] = ft[2]
        emb[2, 4] = ft[3]
        emb[2, 5] = 0.0
        emb[2, 6] = 1.0
        emb[2, 7] = 0.5

        return emb

    def exact_posteriors(self) -> Tuple[float, float]:
        ft = self.factor_table
        p0_is_0 = ft[0] + ft[1]
        p0_is_1 = ft[2] + ft[3]
        p2_is_0 = ft[0] + ft[2]
        p2_is_1 = ft[1] + ft[3]
        z0 = p0_is_0 + p0_is_1
        z2 = p2_is_0 + p2_is_1
        return p0_is_1 / z0, p2_is_1 / z2


def make_graph() -> TwoVarGraph:
    ft = [random.uniform(0.05, 1.0) for _ in range(4)]
    return TwoVarGraph(factor_table=ft)


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

    posteriors = [g.exact_posteriors() for g in graphs]
    Y = torch.zeros(n_graphs, 3)
    for i, (p0, p2) in enumerate(posteriors):
        Y[i, 0] = p0
        Y[i, 1] = 0.5
        Y[i, 2] = p2

    var_mask = X[:, :, 5] == 0

    return X, Y, var_mask