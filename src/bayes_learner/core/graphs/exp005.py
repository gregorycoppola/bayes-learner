"""
Dating graph from QBBN paper (Coppola 2024), Figure 2.

    lonely  ──┐
               f_or ── like_jj ──┐
    exciting──┘                   f_and ── date
                     like_jk ────┘

7 nodes (5 variable, 2 factor):
  Token 0: lonely    — variable, clamped prior, NOT in loss
  Token 1: f_or      — factor
  Token 2: exciting  — variable, clamped prior, NOT in loss
  Token 3: like_jj   — variable, computed by BP, IN loss
  Token 4: like_jk   — variable, clamped prior, NOT in loss
  Token 5: f_and     — factor
  Token 6: date      — variable, computed by BP, IN loss

Only like_jj and date are targets — the transformer must learn to
compute them. The clamped priors are inputs only.

Encoding (d=24):
  [0]  own_belief (init 0.5, or clamped prior for lonely/exciting/like_jk)
  [1]  neighbor_0_index / (n-1)
  [2]  neighbor_1_index / (n-1)   (0 if none)
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
  [13-23] zeros

n=7 nodes, indices 0..6, normalized by 6.
"""
import torch
import random
import time
from dataclasses import dataclass
from typing import Tuple, List

D_MODEL = 24
N_NODES = 7
NORM = 6.0


def _exact_bp_3rounds(
    ft_or: List[float],
    ft_and: List[float],
    p_lonely: float,
    p_exciting: float,
    p_like_jk: float,
) -> Tuple[float, float]:
    """
    Run 3 rounds of BP. Returns (P(like_jj), P(date)) only.
    lonely, exciting, like_jk are clamped — they don't change.
    """
    b_like_jj = 0.5
    b_date    = 0.5

    for _ in range(3):
        # f_or → like_jj: marginalize over both inputs
        m1 = (ft_or[0] * (1-p_lonely) * (1-p_exciting) +
              ft_or[1] * (1-p_lonely) *    p_exciting   +
              ft_or[2] *    p_lonely  * (1-p_exciting)  +
              ft_or[3] *    p_lonely  *    p_exciting)
        m0 = 1.0 - m1
        z  = m0 + m1
        msg_or_to_likejj = m1 / z if z > 0 else 0.5

        # f_and → date: marginalize over both inputs
        m1 = (ft_and[0] * (1-b_like_jj) * (1-p_like_jk) +
              ft_and[1] * (1-b_like_jj) *    p_like_jk  +
              ft_and[2] *    b_like_jj  * (1-p_like_jk) +
              ft_and[3] *    b_like_jj  *    p_like_jk)
        m0 = 1.0 - m1
        z  = m0 + m1
        msg_and_to_date = m1 / z if z > 0 else 0.5

        # Update free variables with flat prior (0.5 prior, updated by msg)
        num = msg_or_to_likejj * 0.5
        den = num + (1 - msg_or_to_likejj) * 0.5
        b_like_jj = num / den if den > 0 else 0.5

        num = msg_and_to_date * 0.5
        den = num + (1 - msg_and_to_date) * 0.5
        b_date = num / den if den > 0 else 0.5

    return b_like_jj, b_date


def _encode_ft(emb: torch.Tensor, row: int, start: int, ft: List[float]):
    emb[row, start]     = ft[0]
    emb[row, start + 1] = ft[1]
    emb[row, start + 2] = ft[2]
    emb[row, start + 3] = ft[3]


@dataclass
class DatingGraph:
    ft_or:      List[float]
    ft_and:     List[float]
    p_lonely:   float
    p_exciting: float
    p_like_jk:  float

    def encode(self) -> torch.Tensor:
        fo, fa = self.ft_or, self.ft_and
        emb = torch.zeros(N_NODES, D_MODEL)

        # Token 0: lonely — clamped prior in dim 0
        emb[0, 0] = self.p_lonely
        emb[0, 1] = 1.0 / NORM
        emb[0, 2] = 0.0
        emb[0, 3] = 0.0
        emb[0, 4] = 0.0 / NORM
        _encode_ft(emb, 0, 5, fo)

        # Token 1: f_or — neighbors lonely(0), exciting(2)
        emb[1, 0] = 0.5
        emb[1, 1] = 0.0 / NORM
        emb[1, 2] = 2.0 / NORM
        emb[1, 3] = 1.0
        emb[1, 4] = 1.0 / NORM
        _encode_ft(emb, 1, 5, fo)

        # Token 2: exciting — clamped prior in dim 0
        emb[2, 0] = self.p_exciting
        emb[2, 1] = 1.0 / NORM
        emb[2, 2] = 0.0
        emb[2, 3] = 0.0
        emb[2, 4] = 2.0 / NORM
        _encode_ft(emb, 2, 5, fo)

        # Token 3: like_jj — two neighbors, gets BOTH tables
        emb[3, 0] = 0.5
        emb[3, 1] = 1.0 / NORM   # neighbor_0 = f_or
        emb[3, 2] = 5.0 / NORM   # neighbor_1 = f_and
        emb[3, 3] = 0.0
        emb[3, 4] = 3.0 / NORM
        _encode_ft(emb, 3, 5, fo)   # left = f_or
        _encode_ft(emb, 3, 9, fa)   # right = f_and

        # Token 4: like_jk — clamped prior in dim 0
        emb[4, 0] = self.p_like_jk
        emb[4, 1] = 5.0 / NORM
        emb[4, 2] = 0.0
        emb[4, 3] = 0.0
        emb[4, 4] = 4.0 / NORM
        _encode_ft(emb, 4, 5, fa)

        # Token 5: f_and — neighbors like_jj(3), like_jk(4)
        emb[5, 0] = 0.5
        emb[5, 1] = 3.0 / NORM
        emb[5, 2] = 4.0 / NORM
        emb[5, 3] = 1.0
        emb[5, 4] = 5.0 / NORM
        _encode_ft(emb, 5, 5, fa)

        # Token 6: date — one neighbor f_and(5)
        emb[6, 0] = 0.5
        emb[6, 1] = 5.0 / NORM
        emb[6, 2] = 0.0
        emb[6, 3] = 0.0
        emb[6, 4] = 6.0 / NORM
        _encode_ft(emb, 6, 5, fa)

        return emb

    def exact_posteriors(self) -> Tuple[float, float]:
        return _exact_bp_3rounds(
            self.ft_or, self.ft_and,
            self.p_lonely, self.p_exciting, self.p_like_jk
        )


def make_graph() -> DatingGraph:
    return DatingGraph(
        ft_or      = [random.uniform(0.05, 1.0) for _ in range(4)],
        ft_and     = [random.uniform(0.05, 1.0) for _ in range(4)],
        p_lonely   = random.uniform(0.05, 0.95),
        p_exciting = random.uniform(0.05, 0.95),
        p_like_jk  = random.uniform(0.05, 0.95),
    )


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

    Y = torch.full((n_graphs, N_NODES), 0.5)
    var_mask = torch.zeros(n_graphs, N_NODES, dtype=torch.bool)
    for i, g in enumerate(graphs):
        ljj, d = g.exact_posteriors()
        Y[i, 3] = ljj   # like_jj
        Y[i, 6] = d     # date
        var_mask[i, 3] = True   # only evaluate on like_jj and date
        var_mask[i, 6] = True

    return X, Y, var_mask