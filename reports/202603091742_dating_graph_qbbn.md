# exp005: Dating Graph from QBBN Paper
**Date:** 2026-03-09
**Experiment:** exp005
**Status:** ✓ STRONG POSITIVE

## Setup

Graph structure from Coppola (2024), Figure 2:

    lonely  ──┐
               f_or ── like_jj ──┐
    exciting──┘                   f_and ── date
                   like_jk ──────┘

This is the exact graph from the QBBN paper's running example. Five variable nodes, two factor nodes. Three rounds of BP required for full propagation: lonely and exciting must flow through f_or to like_jj, and like_jj's updated belief must then flow through f_and to date.

- 20,000 randomly generated graphs
- Factor tables drawn uniformly from [0.05, 1.0]
- Priors for lonely, exciting, like_jk drawn uniformly from [0.05, 0.95]
- Train/val split: 18,000 / 2,000
- Model: BPTransformer, d_in=24, d_model=64, 2 heads, 4 layers, ~206k params
- Loss: MSE on like_jj and date only (the two non-trivial computed nodes)
- Baseline MAE: 0.1413

## Results

| Metric | Value |
|--------|-------|
| Final val MAE | 0.003341 |
| Best val MAE | 0.003271 |
| Baseline MAE (predict 0.5) | 0.141333 |
| Improvement | +97.6% |
| Epochs | 150 |

## Posterior Comparison (final)

| Graph | BP exact | Transformer | Max error |
|-------|----------|-------------|-----------|
| 0 | [0.6014, 0.5760] | [0.6065, 0.5825] | 0.0064 |
| 1 | [0.2793, 0.3758] | [0.2779, 0.3794] | 0.0036 |
| 2 | [0.3392, 0.7418] | [0.3473, 0.7443] | 0.0081 |
| 3 | [0.6253, 0.3524] | [0.6191, 0.3563] | 0.0062 |
| 4 | [0.4949, 0.5833] | [0.4943, 0.5806] | 0.0027 |
| 5 | [0.7676, 0.5645] | [0.7672, 0.5652] | 0.0007 |
| 6 | [0.4625, 0.6446] | [0.4674, 0.6446] | 0.0049 |
| 7 | [0.4956, 0.5051] | [0.5010, 0.5033] | 0.0054 |
| 8 | [0.4508, 0.3429] | [0.4496, 0.3414] | 0.0015 |
| 9 | [0.2974, 0.3695] | [0.2989, 0.3743] | 0.0048 |

## What This Experiment Means

The QBBN paper introduced the dating graph as the central example of probabilistic logical reasoning: given priors on whether jack is lonely and jill is exciting, infer whether jack likes jill, and from that whether they date. The graph encodes a causal theory — loneliness and excitement cause liking, and mutual liking causes dating.

This experiment asks: can a transformer learn to execute that causal reasoning from scratch, on graphs it has never seen, given only the factor tables and priors as input?

The answer is yes, to three decimal places.

The transformer is not memorizing a lookup table. Each of the 2,000 validation graphs has a different factor table and different priors — different theories of how strongly loneliness causes liking, how strongly liking causes dating. The model reads those tables and computes the correct posterior for that specific theory. Graph 5 above shows the tightest result: BP gives [0.7676, 0.5645] and the transformer outputs [0.7672, 0.5652], a max error of 0.0007.

This closes a loop between three pieces of work. The QBBN paper (Coppola 2024) introduced the graphical model and showed belief propagation converges on it empirically. The bp-lean proof showed formally that a transformer can implement BP over such graphs — the weights exist. This experiment shows that gradient descent finds those weights, on the exact graph the QBBN paper was built around.

The transformer has learned to reason about who dates whom.

## Comparison Across Experiments

| Experiment | Graph | Rounds | Final MAE | Improvement |
|------------|-------|--------|-----------|-------------|
| exp001 | v0 --- f1 --- v2 | 1 | 0.000752 | +99.3% |
| exp002 | p1, p2 → AND → dates | 1 | ~0.001 | +99.6% |
| exp003 | 3-var chain (bad encoding) | 2 | 0.068 | +47.9% |
| exp004 | 3-var chain (full encoding) | 2 | 0.002679 | +97.9% |
| exp005 | QBBN dating graph | 3 | 0.003271 | +97.6% |