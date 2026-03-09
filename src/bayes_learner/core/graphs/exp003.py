"""
Chain of 3 variable nodes connected by 2 factor nodes.

    v0 --- f1 --- v1 --- f2 --- v2

Requires 2 rounds of BP for exact posteriors — tests the iterated
transformer loop: encode → transformer → decode → encode → transformer → decode.

Encoding per node (d=8):
  [0] own_belief (init 0.5)
  [1] factor_table[0,0] for primary factor
  [2] factor_table[0,1] for primary factor
  [3] factor_table[1,0] for primary factor
  [4] factor_table[1,1] for primary factor
  [5] node_type (0=variable, 1=factor)
  [6] own_index / (n-1)
  [7] neighbor_0_index / (n-1)

Primary factor assignment:
  v0 → f1's table
  f1 → f1's table
  v1 → f1's table  (left neighbor; f2's info arrives via attention in round 2)
  f2 → f2's table
  v2 → f2's table

n=5 nodes, indices 0..4, normalized by 4.

Neighbor encoding:
  v0:  neighbor_0 = f1 (index 1)
  f1:  neighbor_0 = v0 (index 0), neighbor_1 = v1 (index 2)  — stored in dim 7 and dim 2 override
  v1:  neighbor_0 = f1 (index 1)
  f2:  neighbor_0 = v1 (index 2), neighbor_1 = v2 (index 4)
  v2:  neighbor_0 = f2 (index 3)

Note: for factor nodes we need two neighbor indices.
We use dim 7 for neighbor_0 and repurpose dim 2 for neighbor_1
(factor table dim 2 is less critical since the transformer can
read the full table from dims 1-4).
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
    Run 2 rounds of exact BP on the chain v0 --- f1 --- v1 --- f2 --- v2.
    Returns (P(v0=1), P(v1=1), P(v2=1)).
    """
    # Initial beliefs
    b = [0.5, 0.5, 0.5, 0.5, 0.5]  # v0, f1, v1, f2, v2

    def update_belief(m0, m1):
        num = m0 * m1
        den = num + (1 - m0) * (1 - m1)
        return num / den if den > 0 else 0.5

    def factor_message(ft, b_in):
        """Message from factor to variable given input belief b_in."""
        p1 = ft[2] * b_in + ft[0] * (1 - b_in)  # P(out=1|in) weighted
        p0 = ft[3] * b_in + ft[1] * (1 - b_in)  # wait, let me be precise
        # Factor f connects v_left and v_right
        # Message to v_right given v_left belief b_left:
        # m(v_right=1) = sum_{v_left} f(v_left, v_right=1) * b(v_left)
        # f layout: [f(0,0), f(0,1), f(1,0), f(1,1)]
        m1 = ft[1] * (1 - b_in) + ft[3] * b_in   # f(0,1)*(1-b) + f(1,1)*b
        m0 = ft[0] * (1 - b_in) + ft[2] * b_in   # f(0,0)*(1-b) + f(1,0)*b
        z = m0 + m1
        return m1 / z if z > 0 else 0.5

    # Round 1: messages flow one hop
    # f1 sends to v0 based on v1's belief (0.5)
    # f1 sends to v1 based on v0's belief (0.5)
    # f2 sends to v1 based on v2's belief (0.5)
    # f2 sends to v2 based on v1's belief (0.5)
    msg_f1_to_v0 = factor_message(ft1, b[2])   # f1→v0, using v1 belief
    msg_f1_to_v1 = factor_message(ft1, b[0])   # f1→v1, using v0 belief
    msg_f2_to_v1 = factor_message(ft2, b[4])   # f2→v1, using v2 belief
    msg_f2_to_v2 = factor_message(ft2, b[2])   # f2→v2, using v1 belief

    b[0] = update_belief(msg_f1_to_v0, 0.5)
    b[2] = update_belief(msg_f1_to_v1, msg_f2_to_v1)
    b[4] = update_belief(msg_f2_to_v2, 0.5)

    # Round 2: messages carry updated beliefs
    msg_f1_to_v0 = factor_message(ft1, b[2])
    msg_f1_to_v1 = factor_message(ft1, b[0])
    msg_f2_to_v1 = factor_message(ft2, b[4])
    msg_f2_to_v2 = factor_message(ft2, b[2])

    b[0] = update_belief(msg_f1_to_v0, 0.5)
    b[2] = update_belief(msg_f1_to_v1, msg_f2_to_v1)
    b[4] = update_belief(msg_f2_to_v2, 0.5)

    return b[0], b[2], b[4]


@dataclass
class ChainGraph:
    ft1: List[float]   # factor table for f1
    ft2: List[float]   # factor table for f2

    def encode(self) -> torch.Tensor:
        ft1, ft2 = self.ft1, self.ft2
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: v0 (variable, primary factor: f1, neighbor: f1 at index 1)
        emb[0, 0] = 0.5
        emb[0, 1] = ft1[0]; emb[0, 2] = ft1[1]
        emb[0, 3] = ft1[2]; emb[0, 4] = ft1[3]
        emb[0, 5] = 0.0          # variable
        emb[0, 6] = 0.0 / 4.0   # own index
        emb[0, 7] = 1.0 / 4.0   # neighbor_0 = f1 (index 1)

        # Token 1: f1 (factor, neighbors: v0 at 0, v1 at 2)
        emb[1, 0] = 0.5
        emb[1, 1] = ft1[0]; emb[1, 2] = ft1[1]
        emb[1, 3] = ft1[2]; emb[1, 4] = ft1[3]
        emb[1, 5] = 1.0          # factor
        emb[1, 6] = 1.0 / 4.0   # own index
        emb[1, 7] = 0.0 / 4.0   # neighbor_0 = v0 (index 0)
        # neighbor_1 = v1 (index 2) — stored where? use a separate
        # convention: for factor nodes dim 2 is overloaded as neighbor_1_index
        # after encoding the factor table elsewhere. We accept this impurity
        # and note it for the paper.
        # Actually let's keep it clean and use the reserved slots differently:
        # factor nodes don't need scratch slots the same way.
        # Use dim 4 (normally ft[3]) as neighbor_1 for factor nodes.
        # The transformer can learn to read neighbor indices from dim 7 and dim 4
        # based on node_type.
        emb[1, 4] = 2.0 / 4.0   # neighbor_1 = v1 (index 2) for factor nodes

        # Token 2: v1 (variable, primary factor: f1, neighbors: f1 at 1, f2 at 3)
        emb[2, 0] = 0.5
        emb[2, 1] = ft1[0]; emb[2, 2] = ft1[1]
        emb[2, 3] = ft1[2]; emb[2, 4] = ft1[3]
        emb[2, 5] = 0.0          # variable
        emb[2, 6] = 2.0 / 4.0   # own index
        emb[2, 7] = 1.0 / 4.0   # neighbor_0 = f1 (index 1)
        # neighbor_1 = f2 (index 3) — no clean slot. Use a second pass.
        # For now v1 only explicitly encodes f1 as neighbor. f2's info
        # arrives via f2's token attending to v1 in round 2.

        # Token 3: f2 (factor, neighbors: v1 at 2, v2 at 4)
        emb[3, 0] = 0.5
        emb[3, 1] = ft2[0]; emb[3, 2] = ft2[1]
        emb[3, 3] = ft2[2]; emb[3, 4] = ft2[3]
        emb[3, 5] = 1.0          # factor
        emb[3, 6] = 3.0 / 4.0   # own index
        emb[3, 7] = 2.0 / 4.0   # neighbor_0 = v1 (index 2)
        emb[3, 4] = 4.0 / 4.0   # neighbor_1 = v2 (index 4) — same overload as f1

        # Token 4: v2 (variable, primary factor: f2, neighbor: f2 at index 3)
        emb[4, 0] = 0.5
        emb[4, 1] = ft2[0]; emb[4, 2] = ft2[1]
        emb[4, 3] = ft2[2]; emb[4, 4] = ft2[3]
        emb[4, 5] = 0.0          # variable
        emb[4, 6] = 4.0 / 4.0   # own index
        emb[4, 7] = 3.0 / 4.0   # neighbor_0 = f2 (index 3)

        return emb

    def exact_posteriors(self) -> Tuple[float, float, float]:
        return _exact_bp_2rounds(self.ft1, self.ft2)


def make_graph() -> ChainGraph:
    ft1 = [random.uniform(0.05, 1.0) for _ in range(4)]
    ft2 = [random.uniform(0.05, 1.0) for _ in range(4)]
    return ChainGraph(ft1=ft1, ft2=ft2)


def make_dataset(n_graphs: int, log_every: int = 2000):
    """
    Returns:
      X:        [n_graphs, 5, 8]
      Y:        [n_graphs, 5]
      var_mask: [n_graphs, 5]  True for v0, v1, v2 (tokens 0, 2, 4)
    """
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
    var_mask[:, 0] = True   # v0
    var_mask[:, 2] = True   # v1
    var_mask[:, 4] = True   # v2

    return X, Y, var_mask