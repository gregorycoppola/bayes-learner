"""
AND/OR graph: two variable nodes feeding a hard AND factor, outputting one OR node.

    p1 \
        AND --- dates
    p2 /

This tests whether a transformer can learn to compute the hard AND function:
    P(dates=1) = P(p1=1) * P(p2=1)

This is the simplest non-trivial AND/OR graph from the QBBN dating example.
Exact answer is just multiplication — but the transformer must learn to
branch on node_type to apply different update rules for AND vs OR nodes.

Encoding per node (d=8):
  [0] own_belief (init 0.5)
  [1] neighbor_0_index / (n-1)
  [2] neighbor_1_index / (n-1)  (0.0 if no second neighbor)
  [3] node_type (0=variable, 1=AND factor)
  [4] scratch slot 0
  [5] scratch slot 1
  [6] own_index / (n-1)
  [7] reserved

Nodes:
  Token 0: p1    — variable, neighbor: AND (token 1)
  Token 1: AND   — factor,   neighbors: p1 (token 0), p2 (token 2)
  Token 2: p2    — variable, neighbor: AND (token 1)
  Token 3: dates — variable, neighbor: AND (token 1)

n=4 nodes, indices 0..3, normalized by 3.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import Tuple, List

D_MODEL = 8
N_NODES = 4


@dataclass
class AndOrGraph:
    """
    p1, p2 are independent priors.
    dates = p1 * p2  (hard AND).
    """
    p1: float
    p2: float

    def encode(self) -> torch.Tensor:
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: p1 (variable, neighbor: AND at index 1)
        emb[0, 0] = 0.5          # belief
        emb[0, 1] = 1.0 / 3.0   # neighbor 0 = AND (index 1) / 3
        emb[0, 2] = 0.0          # no second neighbor
        emb[0, 3] = 0.0          # variable node
        emb[0, 4] = 0.0          # scratch 0
        emb[0, 5] = 0.0          # scratch 1
        emb[0, 6] = 0.0 / 3.0   # own index 0 / 3
        emb[0, 7] = self.p1      # encode prior directly in reserved slot

        # Token 1: AND factor (neighbors: p1 at 0, p2 at 2)
        emb[1, 0] = 0.5
        emb[1, 1] = 0.0 / 3.0   # neighbor 0 = p1 (index 0) / 3
        emb[1, 2] = 2.0 / 3.0   # neighbor 1 = p2 (index 2) / 3
        emb[1, 3] = 1.0          # AND factor node
        emb[1, 4] = 0.0
        emb[1, 5] = 0.0
        emb[1, 6] = 1.0 / 3.0   # own index 1 / 3
        emb[1, 7] = 0.0

        # Token 2: p2 (variable, neighbor: AND at index 1)
        emb[2, 0] = 0.5
        emb[2, 1] = 1.0 / 3.0   # neighbor 0 = AND (index 1) / 3
        emb[2, 2] = 0.0
        emb[2, 3] = 0.0          # variable node
        emb[2, 4] = 0.0
        emb[2, 5] = 0.0
        emb[2, 6] = 2.0 / 3.0   # own index 2 / 3
        emb[2, 7] = self.p2      # encode prior directly in reserved slot

        # Token 3: dates (variable, neighbor: AND at index 1)
        emb[3, 0] = 0.5
        emb[3, 1] = 1.0 / 3.0   # neighbor 0 = AND (index 1) / 3
        emb[3, 2] = 0.0
        emb[3, 3] = 0.0          # variable node
        emb[3, 4] = 0.0
        emb[3, 5] = 0.0
        emb[3, 6] = 3.0 / 3.0   # own index 3 / 3
        emb[3, 7] = 0.0

        return emb

    def exact_posteriors(self) -> Tuple[float, float, float, float]:
        """
        Returns (p1, p2, and_belief, dates).
        p1 and p2 are observed priors.
        and_belief = p1 * p2  (hard AND message to dates).
        dates = p1 * p2.
        """
        and_belief = self.p1 * self.p2
        return self.p1, self.p2, and_belief, and_belief


def make_graph() -> AndOrGraph:
    p1 = random.uniform(0.05, 0.95)
    p2 = random.uniform(0.05, 0.95)
    return AndOrGraph(p1=p1, p2=p2)


def make_dataset(n_graphs: int, log_every: int = 2000):
    """
    Returns:
      X:        [n_graphs, 4, 8]
      Y:        [n_graphs, 4]   posteriors per node
      var_mask: [n_graphs, 4]   True for variable nodes (node_type == 0)

    Y layout per graph:
      [0] = p1        (observed prior — clamped, not really predicted)
      [1] = 0.5       (AND factor node — masked out)
      [2] = p2        (observed prior — clamped)
      [3] = p1 * p2   (dates — the target we care about)

    We only evaluate loss on token 3 (dates).
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
        p1, p2, and_b, dates = g.exact_posteriors()
        Y[i, 0] = p1
        Y[i, 1] = 0.5    # AND node masked
        Y[i, 2] = p2
        Y[i, 3] = dates

    # Only evaluate on dates (token 3) — that's the output we care about
    # p1 and p2 are inputs, AND is a factor
    var_mask = torch.zeros(n_graphs, N_NODES, dtype=torch.bool)
    var_mask[:, 3] = True   # only dates is the target

    return X, Y, var_mask