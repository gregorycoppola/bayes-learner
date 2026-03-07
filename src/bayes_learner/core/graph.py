"""Random tree-structured factor graph generation and exact BP."""
import torch
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

K = 2  # neighbors per node, matches Preliminaries.lean

@dataclass
class FactorGraph:
    """
    Tree-structured factor graph with K=2 neighbors per node.
    Nodes alternate variable/factor in bipartite structure.
    n must be odd: variable nodes at even indices, factor nodes at odd.
    """
    n: int
    node_type: List[int]        # 0=variable, 1=factor
    neighbors: List[List[int]]  # neighbors[j] = [nb0, nb1]
    factor_table: List[List[float]]  # factor_table[j] = 4 floats for factor nodes

    def to_tensors(self):
        """
        Encode as transformer input matching encodeBPState from Preliminaries.lean.
        Returns embedding tensor of shape [n, 8].

        dim 0 = belief (uniform init = 0.5)
        dim 1 = neighbor 0 index (float)
        dim 2 = neighbor 1 index (float)
        dim 3 = node type (0=variable, 1=factor)
        dim 4 = scratch 0 (0.0)
        dim 5 = scratch 1 (0.0)
        dim 6,7 = unused (0.0)
        """
        D = 8
        emb = torch.zeros(self.n, D)
        for j in range(self.n):
            emb[j, 0] = 0.5  # uniform belief
            emb[j, 1] = float(self.neighbors[j][0])
            emb[j, 2] = float(self.neighbors[j][1])
            emb[j, 3] = float(self.node_type[j])
        return emb


def make_tree(n_vars: int) -> FactorGraph:
    """
    Build a random binary tree factor graph.
    n_vars variable nodes, n_vars-1 factor nodes (for a tree).
    Total nodes = 2*n_vars - 1.
    Variable nodes at even indices 0,2,4,...
    Factor nodes at odd indices 1,3,5,...

    Each factor node connects exactly 2 variable nodes.
    Each variable node connects to up to 2 factor nodes (padded with self-index if leaf).
    """
    assert n_vars >= 2
    n = 2 * n_vars - 1
    node_type = [0 if i % 2 == 0 else 1 for i in range(n)]

    # Build a random spanning tree over variable nodes via random factor connections
    # Factor node f_i (at index 2i+1) connects variable v_i (2i) and v_{i+1} (2i+2)
    # for a chain topology — simplest valid tree
    var_factor_neighbors: List[List[int]] = [[] for _ in range(n)]
    factor_table = [[1.0] * 4 for _ in range(n)]

    for fi in range(n_vars - 1):
        f_idx = 2 * fi + 1
        v0 = 2 * fi
        v1 = 2 * fi + 2
        var_factor_neighbors[f_idx] = [v0, v1]
        var_factor_neighbors[v0].append(f_idx)
        var_factor_neighbors[v1].append(f_idx)
        # Random factor table (positive entries)
        factor_table[f_idx] = [random.uniform(0.1, 1.0) for _ in range(4)]

    # Pad all neighbor lists to exactly K=2 (use self-index for leaves)
    neighbors = []
    for j in range(n):
        nb = var_factor_neighbors[j][:K]
        while len(nb) < K:
            nb = nb + [j]  # pad with self
        neighbors.append(nb)

    return FactorGraph(n=n, node_type=node_type, neighbors=neighbors, factor_table=factor_table)


def run_bp(fg: FactorGraph, n_iters: int = 50) -> List[float]:
    """
    Run sum-product belief propagation on the factor graph.
    Returns exact marginal beliefs for all variable nodes.
    On trees, converges exactly in diameter rounds.
    """
    n = fg.n
    # Messages: msg[i][j] = message from node i to node j
    msg = [[1.0] * 2 for _ in range(n * n)]  # msg[i*n+j][0/1]

    def get_msg(i, j):
        return msg[i * n + j]

    def set_msg(i, j, m):
        msg[i * n + j] = m

    for _ in range(n_iters):
        new_msg = [[1.0] * 2 for _ in range(n * n)]
        for j in range(n):
            for nb_idx, i in enumerate(fg.neighbors[j]):
                if i == j:
                    continue
                if fg.node_type[j] == 0:
                    # variable → factor: product of incoming messages from other neighbors
                    m = [1.0, 1.0]
                    for other in fg.neighbors[j]:
                        if other == i or other == j:
                            continue
                        inc = get_msg(other, j)
                        m[0] *= inc[0]
                        m[1] *= inc[1]
                    new_msg[j * n + i] = m
                else:
                    # factor → variable: sum over factor table
                    other_vars = [v for v in fg.neighbors[j] if v != i and v != j]
                    m = [0.0, 0.0]
                    ft = fg.factor_table[j]
                    for xi in range(2):
                        for xo in range(2):
                            # ft indexed as [xi * 2 + xo] where xi=outgoing var value
                            inc = get_msg(other_vars[0], j) if other_vars else [1.0, 1.0]
                            m[xi] += ft[xi * 2 + xo] * inc[xo]
                    new_msg[j * n + i] = m

        msg = new_msg

    # Compute beliefs
    beliefs = []
    for j in range(n):
        if fg.node_type[j] == 0:
            b = [1.0, 1.0]
            for i in fg.neighbors[j]:
                if i == j:
                    continue
                inc = get_msg(i, j)
                b[0] *= inc[0]
                b[1] *= inc[1]
            Z = b[0] + b[1]
            beliefs.append(b[1] / Z if Z > 0 else 0.5)
        else:
            beliefs.append(0.5)  # factor nodes: not evaluated
    return beliefs


def make_dataset(n_graphs: int, n_vars: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n_graphs random factor graphs.
    Returns:
      X: [n_graphs, n, 8] — encoded initial states
      Y: [n_graphs, n]    — exact BP posteriors per node
    """
    graphs = [make_tree(n_vars) for _ in range(n_graphs)]
    X = torch.stack([g.to_tensors() for g in graphs])
    Y = torch.tensor([[run_bp(g)[j] for j in range(g.n)] for g in graphs])
    return X, Y