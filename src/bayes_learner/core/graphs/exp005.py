"""
Dating graph from QBBN paper (Coppola 2024), Figure 2.

    lonely  ──┐
               f_or ── like_jj ──┐
    exciting──┘                   f_and ── date
                     like_jk ────┘

7 nodes (5 variable, 2 factor):
  Token 0: lonely    — variable, one neighbor: f_or (1)
  Token 1: f_or      — factor,   neighbors: lonely (0), exciting (2)
  Token 2: exciting  — variable, one neighbor: f_or (1)
  Token 3: like_jj   — variable, two neighbors: f_or (1), f_and (5)
  Token 4: like_jk   — variable, one neighbor: f_and (5)
  Token 5: f_and     — factor,   neighbors: like_jj (3), like_jk (4)
  Token 6: date      — variable, one neighbor: f_and (5)

3 rounds of BP needed for full propagation.

Encoding (d=24):
  [0]  own_belief (init 0.5)
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


def _factor_msg(ft: List[float], b_in: float) -> float:
    """
    Message from a 2-input factor to one output variable,
    marginalizing over the other input with current belief b_in.
    ft layout: [f(0,0), f(0,1), f(1,0), f(1,1)]
    where first index = input var, second index = output var.
    m(out=1) = ft(in=0,out=1)*(1-b_in) + ft(in=1,out=1)*b_in
    """
    m1 = ft[1] * (1 - b_in) + ft[3] * b_in
    m0 = ft[0] * (1 - b_in) + ft[2] * b_in
    z = m0 + m1
    return m1 / z if z > 0 else 0.5


def _update_belief(prior: float, msg: float) -> float:
    num = msg * prior
    den = num + (1 - msg) * (1 - prior)
    return num / den if den > 0 else 0.5


def _exact_bp_3rounds(
    ft_or: List[float],
    ft_and: List[float],
    p_lonely: float,
    p_exciting: float,
    p_like_jk: float,
) -> Tuple[float, float, float, float, float]:
    """
    Run 3 rounds of BP on the dating graph.

    Clamped nodes (fixed priors): lonely, exciting, like_jk
    Free nodes (updated by BP):   like_jj, date

    Round 1: f_or sees lonely=p_lonely, exciting=p_exciting → msg to like_jj
             f_and sees like_jj=0.5, like_jk=p_like_jk → msg to date
    Round 2: f_and sees updated like_jj → better msg to date
    Round 3: converge
    """
    b_lonely   = p_lonely
    b_exciting = p_exciting
    b_like_jk  = p_like_jk
    b_like_jj  = 0.5
    b_date     = 0.5

    for _ in range(3):
        # f_or → like_jj: marginalizes over lonely (using exciting belief)
        #                  and over exciting (using lonely belief)
        # Two-input factor: message to output = sum over both inputs
        # m(like_jj=1) = sum_{l,e} ft_or(l,e,like_jj=1) * b(l) * b(e)
        # With ft layout [f(0,0), f(0,1), f(1,0), f(1,1)] where
        # first index = lonely, second = exciting... actually for 2-input
        # factors we need to marginalize over BOTH inputs:
        m_or_to_likejj_1 = (
            ft_or[0] * (1-b_lonely) * (1-b_exciting) +  # f(l=0,e=0) * ...
            ft_or[1] * (1-b_lonely) *    b_exciting  +  # f(l=0,e=1) * ...
            ft_or[2] *    b_lonely  * (1-b_exciting) +  # f(l=1,e=0) * ...
            ft_or[3] *    b_lonely  *    b_exciting      # f(l=1,e=1) * ...
        )
        m_or_to_likejj_0 = 1.0 - m_or_to_likejj_1
        z = m_or_to_likejj_0 + m_or_to_likejj_1
        msg_or_to_likejj = m_or_to_likejj_1 / z if z > 0 else 0.5

        # f_and → date: marginalizes over like_jj and like_jk
        m_and_to_date_1 = (
            ft_and[0] * (1-b_like_jj) * (1-b_like_jk) +
            ft_and[1] * (1-b_like_jj) *    b_like_jk  +
            ft_and[2] *    b_like_jj  * (1-b_like_jk) +
            ft_and[3] *    b_like_jj  *    b_like_jk
        )
        m_and_to_date_0 = 1.0 - m_and_to_date_1
        z = m_and_to_date_0 + m_and_to_date_1
        msg_and_to_date = m_and_to_date_1 / z if z > 0 else 0.5

        # Update free variables
        b_like_jj = _update_belief(0.5, msg_or_to_likejj)
        b_date    = _update_belief(0.5, msg_and_to_date)
        # clamped nodes don't change

    return b_lonely, b_exciting, b_like_jj, b_like_jk, b_date


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

    def exact_posteriors(self) -> Tuple[float, float, float, float, float]:
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
        pl, pe, ljj, ljk, d = g.exact_posteriors()
        Y[i, 0] = pl
        Y[i, 2] = pe
        Y[i, 3] = ljj
        Y[i, 4] = ljk
        Y[i, 6] = d
        var_mask[i, 0] = True
        var_mask[i, 2] = True
        var_mask[i, 3] = True
        var_mask[i, 4] = True
        var_mask[i, 6] = True

    return X, Y, var_mask