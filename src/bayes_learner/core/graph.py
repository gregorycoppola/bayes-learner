"""
Random tree-structured factor graph generation and exact BP.

Encoding (updated to match what the attention construction actually needs):
  dim 0 = belief (init 0.5)
  dim 1 = neighbor 0 index (float) — query for head 0
  dim 2 = neighbor 1 index (float) — query for head 1
  dim 3 = node type (0=variable, 1=factor)
  dim 4 = scratch 0 (0.0)
  dim 5 = scratch 1 (0.0)
  dim 6 = own index (float) — key for both heads
  dim 7 = unused (0.0)

Head 0 attention: node j queries with dim1 (neighbor0 index),
                  node k keys with dim6 (own index).
                  Score peaks when k's own index == j's neighbor0 index.
                  This correctly routes to neighbor 0.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import List

K = 2
D_MODEL = 8


@dataclass
class FactorGraph:
    n: int
    node_type: List[int]
    neighbors: List[List[int]]
    factor_table: List[List[float]]

    def encode(self) -> torch.Tensor:
        emb = torch.zeros(self.n, D_MODEL)
        for j in range(self.n):
            emb[j, 0] = 0.5                          # belief
            emb[j, 1] = float(self.neighbors[j][0])  # neighbor 0 index (query)
            emb[j, 2] = float(self.neighbors[j][1])  # neighbor 1 index (query)
            emb[j, 3] = float(self.node_type[j])     # node type
            # dim 4,5 = scratch (0.0)
            emb[j, 6] = float(j)                     # own index (key)
        return emb


def make_chain(n_vars: int) -> FactorGraph:
    assert n_vars >= 2
    n = 2 * n_vars - 1
    node_type = [0 if i % 2 == 0 else 1 for i in range(n)]
    var_neighbors: List[List[int]] = [[] for _ in range(n)]
    factor_table = [[1.0] * 4 for _ in range(n)]

    for fi in range(n_vars - 1):
        f_idx = 2 * fi + 1
        v0    = 2 * fi
        v1    = 2 * fi + 2
        var_neighbors[f_idx] = [v0, v1]
        var_neighbors[v0].append(f_idx)
        var_neighbors[v1].append(f_idx)
        factor_table[f_idx] = [random.uniform(0.1, 1.0) for _ in range(4)]

    neighbors = []
    for j in range(n):
        nb = var_neighbors[j][:K]
        while len(nb) < K:
            nb = nb + [j]
        neighbors.append(nb)

    return FactorGraph(n=n, node_type=node_type,
                       neighbors=neighbors, factor_table=factor_table)


def run_bp(fg: FactorGraph, n_iters: int = 50) -> List[float]:
    n = fg.n
    msg = {(i, j): [1.0, 1.0]
           for i in range(n)
           for j in fg.neighbors[i] if j != i}

    for _ in range(n_iters):
        new_msg = {}
        for j in range(n):
            for i in fg.neighbors[j]:
                if i == j:
                    continue
                if fg.node_type[j] == 0:
                    m = [1.0, 1.0]
                    for other in fg.neighbors[j]:
                        if other == i or other == j:
                            continue
                        if (other, j) in msg:
                            m[0] *= msg[(other, j)][0]
                            m[1] *= msg[(other, j)][1]
                    new_msg[(j, i)] = m
                else:
                    other_vars = [v for v in fg.neighbors[j]
                                  if v != i and v != j]
                    ft = fg.factor_table[j]
                    m = [0.0, 0.0]
                    for xi in range(2):
                        for xo in range(2):
                            inc = (msg.get((other_vars[0], j), [1.0, 1.0])
                                   if other_vars else [1.0, 1.0])
                            m[xi] += ft[xi * 2 + xo] * inc[xo]
                    new_msg[(j, i)] = m
        msg.update(new_msg)

    beliefs = []
    for j in range(n):
        if fg.node_type[j] == 0:
            b = [1.0, 1.0]
            for i in fg.neighbors[j]:
                if i == j:
                    continue
                if (i, j) in msg:
                    b[0] *= msg[(i, j)][0]
                    b[1] *= msg[(i, j)][1]
            Z = b[0] + b[1]
            beliefs.append(b[1] / Z if Z > 1e-10 else 0.5)
        else:
            beliefs.append(0.5)
    return beliefs


def make_dataset(n_graphs: int, n_vars: int = 5, log_every: int = 1000):
    t0 = time.time()
    graphs = []
    for i in range(n_graphs):
        graphs.append(make_chain(n_vars))
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_graphs - i - 1) / rate
            print(f"[DATA]   {i+1}/{n_graphs} graphs "
                  f"({rate:.0f}/s, ~{remaining:.0f}s remaining)")

    X = torch.stack([g.encode() for g in graphs])
    Y = torch.tensor([[run_bp(g)[j] for j in range(g.n)]
                      for g in graphs], dtype=torch.float32)
    var_mask = (X[:, :, 3] == 0)
    return X, Y, var_mask