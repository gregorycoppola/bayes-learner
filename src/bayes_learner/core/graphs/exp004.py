"""
Tree graph: two variables feeding a factor, chaining to another variable.

    v0 \
        f1 --- v2 --- f2 --- v3
    v1 /

v2 is the key node — it has TWO factor neighbors (f1 and f2).
This is the first experiment testing the full two-head gathering mechanism
on a variable node that needs both neighbors simultaneously.

Exact BP converges in 2 rounds on this tree (diameter = 4).

Encoding per node (d=8):
  [0] own_belief (init 0.5)
  [1] neighbor_0_index / (n-1)
  [2] neighbor_1_index / (n-1)  (0.0 if no second neighbor)
  [3] node_type (0=variable, 1=factor)
  [4] factor_table entry 0  (f(0,0)) — own factor if variable, own table if factor
  [5] factor_table entry 1  (f(0,1))
  [6] own_index / (n-1)
  [7] factor_table entry 2  (f(1,0))
  NOTE: f(1,1) is dropped — only 3 entries fit cleanly with 2 neighbor slots
  Actually let's use a different layout to fit everything:

D_MODEL = 10 for this experiment — we need more dims.

Revised encoding (d=10):
  [0] own_belief (init 0.5)
  [1] neighbor_0_index / (n-1)
  [2] neighbor_1_index / (n-1)
  [3] node_type (0=variable, 1=factor)
  [4] ft[0,0]
  [5] ft[0,1]
  [6] ft[1,0]
  [7] ft[1,1]
  [8] own_index / (n-1)
  [9] scratch (reserved)

Each node carries the factor table of its PRIMARY factor neighbor.
v0: f1's table
v1: f1's table
v2: f1's table (left factor — f2's info comes via attention)
f1: f1's table
f2: f2's table
v3: f2's table

Nodes (5 tokens, indices 0..4, normalized by 4):
  Token 0: v0  — variable, neighbors: [f1]
  Token 1: v1  — variable, neighbors: [f1]
  Token 2: f1  — factor,   neighbors: [v0, v1, v2]  (arity 3 — wait...)

Hmm — f1 connects v0, v1 to v2. That's a 3-way factor.
Actually let's re-read the graph:

    v0 \
        f1 --- v2 --- f2 --- v3
    v1 /

f1 is a pairwise factor between {v0,v1} combined and v2.
In QBBN terms f1 is an AND gate: takes v0 AND v1, outputs to v2.
f2 is a pairwise factor between v2 and v3.

So f1 has 3 neighbors (v0, v1, v2) and f2 has 2 neighbors (v2, v3).

For simplicity let's make f1 a standard pairwise factor between
a single input and v2 — i.e. treat {v0,v1} as two SEPARATE factors:

    v0 --- fa --- v2 --- fb --- v3
    v1 --- fc /

Where fa and fc both connect into v2. So v2 has neighbors [fa, fc, fb].
That's 3 neighbors — beyond K=2.

Simplest fix: make f1 take only ONE of v0/v1 directly and let's just do:

    v0 --- f1 \
                v2 --- f3 --- v3
    v1 --- f2 /

5 nodes: v0, v1, v2, v3, f1, f2, f3 = 7 nodes total.
v2 has neighbors f1, f2, f3 — still 3 neighbors.

OK let's just go with the clean version that matches K=2:

    v0 --- f1 --- v2 --- f2 --- v3

And separately have v1 as a second input to f1 making it:

Actually the simplest graph where a variable has exactly 2 factor
neighbors and K=2 is maintained throughout is:

    v0 --- f1 --- v1 --- f2 --- v2

Which is just exp003. The issue is that to have a variable with 2
factor neighbors AND keep K=2 everywhere, we need a chain.

So the real new thing here is: v1 in exp003 already HAS two factor
neighbors. We just didn't encode it cleanly. Let's fix that encoding
and re-run, making v1's second neighbor explicit.

n=5 nodes. Indices 0..4 normalized by 4.
  Token 0: v0  — variable, neighbors: [f1 at 1]
  Token 1: f1  — factor,   neighbors: [v0 at 0, v1 at 2]
  Token 2: v1  — variable, neighbors: [f1 at 1, f2 at 3]  ← TWO factor neighbors
  Token 3: f2  — factor,   neighbors: [v1 at 2, v2 at 4]
  Token 4: v2  — variable, neighbors: [f2 at 3]

This IS exp003 but with v1 getting BOTH neighbor indices explicitly.
The difference: in exp003 v1 only had neighbor_0=f1. Now it gets
neighbor_0=f1 AND neighbor_1=f2 in dims 1 and 2.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import Tuple, List

D_MODEL = 8
N_NODES = 5


def _exact_bp_2rounds(ft1: List[float], ft2: List[float]) -> Tuple[float, float, float]:
    """
    Run 2 rounds of exact BP on v0 --- f1 --- v1 --- f2 --- v2.
    Returns (P(v0=1), P(v1=1), P(v2=1)).
    """
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
    ft1: List[float]
    ft2: List[float]

    def encode(self) -> torch.Tensor:
        ft1, ft2 = self.ft1, self.ft2
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: v0 — neighbor_0=f1(1), no second neighbor
        emb[0, 0] = 0.5
        emb[0, 1] = 1.0 / 4.0   # neighbor_0 = f1 at index 1
        emb[0, 2] = 0.0          # no second neighbor
        emb[0, 3] = 0.0          # variable
        emb[0, 4] = ft1[0]; emb[0, 5] = ft1[1]
        emb[0, 6] = 0.0 / 4.0   # own index
        emb[0, 7] = ft1[2]       # ft[1,0] in dim 7

        # Token 1: f1 — neighbors: v0(0), v1(2)
        emb[1, 0] = 0.5
        emb[1, 1] = 0.0 / 4.0   # neighbor_0 = v0 at index 0
        emb[1, 2] = 2.0 / 4.0   # neighbor_1 = v1 at index 2
        emb[1, 3] = 1.0          # factor
        emb[1, 4] = ft1[0]; emb[1, 5] = ft1[1]
        emb[1, 6] = 1.0 / 4.0   # own index
        emb[1, 7] = ft1[2]

        # Token 2: v1 — neighbor_0=f1(1), neighbor_1=f2(3)  ← KEY NODE
        emb[2, 0] = 0.5
        emb[2, 1] = 1.0 / 4.0   # neighbor_0 = f1 at index 1
        emb[2, 2] = 3.0 / 4.0   # neighbor_1 = f2 at index 3  ← explicit now
        emb[2, 3] = 0.0          # variable
        emb[2, 4] = ft1[0]; emb[2, 5] = ft1[1]
        emb[2, 6] = 2.0 / 4.0   # own index
        emb[2, 7] = ft1[2]

        # Token 3: f2 — neighbors: v1(2), v2(4)
        emb[3, 0] = 0.5
        emb[3, 1] = 2.0 / 4.0   # neighbor_0 = v1 at index 2
        emb[3, 2] = 4.0 / 4.0   # neighbor_1 = v2 at index 4
        emb[3, 3] = 1.0          # factor
        emb[3, 4] = ft2[0]; emb[3, 5] = ft2[1]
        emb[3, 6] = 3.0 / 4.0   # own index
        emb[3, 7] = ft2[2]

        # Token 4: v2 — neighbor_0=f2(3), no second neighbor
        emb[4, 0] = 0.5
        emb[4, 1] = 3.0 / 4.0   # neighbor_0 = f2 at index 3
        emb[4, 2] = 0.0          # no second neighbor
        emb[4, 3] = 0.0          # variable
        emb[4, 4] = ft2[0]; emb[4, 5] = ft2[1]
        emb[4, 6] = 4.0 / 4.0   # own index
        emb[4, 7] = ft2[2]

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
        Y[i, 1] = 0.5   # f1 masked
        Y[i, 2] = v1
        Y[i, 3] = 0.5   # f2 masked
        Y[i, 4] = v2

    var_mask = torch.zeros(n_graphs, N_NODES, dtype=torch.bool)
    var_mask[:, 0] = True
    var_mask[:, 2] = True
    var_mask[:, 4] = True

    return X, Y, var_mask