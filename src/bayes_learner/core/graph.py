"""
Random tree-structured factor graph generation and exact BP.

Encoding matches encodeBPState from transformer-bp-lean/Preliminaries.lean:
  dim 0 = belief (init 0.5)
  dim 1 = neighbor 0 index (float)
  dim 2 = neighbor 1 index (float)
  dim 3 = node type (0=variable, 1=factor)
  dim 4 = scratch 0 (0.0)
  dim 5 = scratch 1 (0.0)
  dim 6,7 = unused (0.0)

Chain topology: var-fac-var-fac-var-fac-var-fac-var
  variable nodes at even indices: 0,2,4,6,8
  factor nodes at odd indices:    1,3,5,7
"""
import torch
import random
from dataclasses import dataclass
from typing import List


K = 2       # neighbors per node
D_MODEL = 8 # embedding dimension


@dataclass
class FactorGraph:
    n: int
    node_type: List[int]         # 0=variable, 1=factor
    neighbors: List[List[int]]   # neighbors[j] = [nb0, nb1]
    factor_table: List[List[float]]  # 4 entries per factor node

    def encode(self) -> torch.Tensor:
        """
        Returns [n, 8] tensor matching encodeBPState.
        """
        emb = torch.zeros(self.n, D_MODEL)
        for j in range(self.n):
            emb[j, 0] = 0.5                          # uniform belief
            emb[j, 1] = float(self.neighbors[j][0])  # neighbor 0 index
            emb[j, 2] = float(self.neighbors[j][1])  # neighbor 1 index
            emb[j, 3] = float(self.node_type[j])     # 0=var, 1=fac
            # dims 4,5,6,7 = 0.0 (scratch slots, unused)
        return emb


def make_chain(n_vars: int) -> FactorGraph:
    """
    Chain topology: v0 - f1 - v2 - f3 - v4 - ... 
    n_vars variable nodes, n_vars-1 factor nodes.
    Total nodes = 2*n_vars - 1.

    Neighbor padding: leaf variable nodes have only 1 factor neighbor,
    padded with self-index to fill K=2 slots.
    Factor nodes always have exactly 2 variable neighbors.
    """
    assert n_vars >= 2
    n = 2 * n_vars - 1
    node_type = [0 if i % 2 == 0 else 1 for i in range(n)]

    # Build neighbor lists
    var_neighbors: List[List[int]] = [[] for _ in range(n)]
    factor_table = [[1.0] * 4 for _ in range(n)]

    for fi in range(n_vars - 1):
        f_idx = 2 * fi + 1
        v0    = 2 * fi
        v1    = 2 * fi + 2
        var_neighbors[f_idx] = [v0, v1]
        var_neighbors[v0].append(f_idx)
        var_neighbors[v1].append(f_idx)
        # Random positive factor table: f(x0, x1) for x0,x1 in {0,1}
        # indexed as [x0*2 + x1]
        factor_table[f_idx] = [random.uniform(0.1, 1.0) for _ in range(4)]

    # Pad neighbor lists to exactly K=2
    neighbors = []
    for j in range(n):
        nb = var_neighbors[j][:K]
        while len(nb) < K:
            nb = nb + [j]  # pad leaf with self-index
        neighbors.append(nb)

    return FactorGraph(n=n, node_type=node_type,
                       neighbors=neighbors, factor_table=factor_table)


def run_bp(fg: FactorGraph, n_iters: int = 50) -> List[float]:
    """
    Sum-product belief propagation on a tree factor graph.
    Converges exactly in diameter rounds; 50 iters is conservative.
    Returns belief P(x=1) for each node (0.5 for factor nodes).
    """
    n = fg.n
    # msg[i][j] = message from node i to node j, as [m(x=0), m(x=1)]
    msg = {(i, j): [1.0, 1.0]
           for i in range(n)
           for j in fg.neighbors[i] if j != i}

    for iteration in range(n_iters):
        new_msg = {}
        for j in range(n):
            for i in fg.neighbors[j]:
                if i == j:
                    continue
                if fg.node_type[j] == 0:
                    # variable → factor: product of messages from other neighbors
                    m = [1.0, 1.0]
                    for other in fg.neighbors[j]:
                        if other == i or other == j:
                            continue
                        if (other, j) in msg:
                            m[0] *= msg[(other, j)][0]
                            m[1] *= msg[(other, j)][1]
                    new_msg[(j, i)] = m
                else:
                    # factor → variable: marginalize over factor table
                    other_vars = [v for v in fg.neighbors[j]
                                  if v != i and v != j]
                    ft = fg.factor_table[j]
                    m = [0.0, 0.0]
                    for xi in range(2):   # value at node i
                        for xo in range(2):  # value at other var
                            inc = msg.get((other_vars[0], j),
                                          [1.0, 1.0]) if other_vars else [1.0, 1.0]
                            m[xi] += ft[xi * 2 + xo] * inc[xo]
                    new_msg[(j, i)] = m
        msg.update(new_msg)

    # Compute marginal beliefs
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


def make_dataset(n_graphs: int, n_vars: int = 5):
    """
    Returns:
      X:        [n_graphs, n, 8]  encoded initial states
      Y:        [n_graphs, n]     exact BP posteriors
      var_mask: [n_graphs, n]     True = variable node
    """
    graphs = [make_chain(n_vars) for _ in range(n_graphs)]
    X = torch.stack([g.encode() for g in graphs])
    Y = torch.tensor([[run_bp(g)[j] for j in range(g.n)]
                      for g in graphs], dtype=torch.float32)
    var_mask = (X[:, :, 3] == 0)
    return X, Y, var_mask