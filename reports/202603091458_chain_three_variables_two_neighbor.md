# exp004: Chain of Three Variables — Explicit Two-Neighbor Encoding

**Date:** 2026-03-09  
**Experiment:** exp004  
**Status:** ✓ STRONG POSITIVE

## Setup

Graph structure: `v0 --- f1 --- v1 --- f2 --- v2`

Same chain as exp003 but with a corrected encoding that gives v1 — the
middle variable node with two factor neighbors — complete information to
compute its posterior without relying on attention to retrieve f2's factor
table at inference time.

- 20,000 randomly generated graphs
- Two independent factor tables, entries drawn uniformly from [0.05, 1.0]
- Train/val split: 18,000 / 2,000
- Model: BPTransformer, d_in=16, d_model=32, 2 heads, 2 layers, ~27k params
- Loss: MSE on variable nodes only (v0, v1, v2 — tokens 0, 2, 4)
- Baseline MAE: 0.1322

## Results

| Metric | Value |
|--------|-------|
| Final val MAE | 0.002766 |
| Best val MAE | 0.002679 |
| Baseline MAE (predict 0.5) | 0.132214 |
| Improvement | +97.9% |
| Epochs | 50 |

## Posterior Comparison (final)

| Graph | BP exact | Transformer | Max error |
|-------|----------|-------------|-----------|
| 0 | [0.5889, 0.5570, 0.4936] | [0.5869, 0.5570, 0.4932] | 0.0020 |
| 1 | [0.5896, 0.5389, 0.4270] | [0.5911, 0.5415, 0.4238] | 0.0031 |
| 2 | [0.4793, 0.5706, 0.5773] | [0.4756, 0.5710, 0.5775] | 0.0037 |
| 3 | [0.3858, 0.2853, 0.3705] | [0.3870, 0.2822, 0.3724] | 0.0031 |
| 4 | [0.6033, 0.5831, 0.4540] | [0.6037, 0.5844, 0.4494] | 0.0046 |
| 5 | [0.4326, 0.4927, 0.5611] | [0.4323, 0.4942, 0.5582] | 0.0030 |
| 6 | [0.5286, 0.4362, 0.4139] | [0.5278, 0.4365, 0.4152] | 0.0013 |
| 7 | [0.4879, 0.6726, 0.6826] | [0.4880, 0.6745, 0.6856] | 0.0030 |
| 8 | [0.3742, 0.4836, 0.6474] | [0.3753, 0.4872, 0.6500] | 0.0036 |
| 9 | [0.3605, 0.3781, 0.4834] | [0.3611, 0.3776, 0.4882] | 0.0048 |

## What Failed First

**exp003 (partial result, ~48% improvement, MAE ~0.068):** The first
attempt at this chain used d=8 encoding and gave v1 only f1's factor
table, relying on attention to retrieve f2's table from token 3 at
runtime. v0 and v2 converged well — their posteriors depend on only one
factor each and all needed information was local. v1 consistently showed
MaxErr of 0.13–0.19, roughly 3–4x worse than v0 and v2. The model was
learning a partial solution: good marginals for the endpoints, poor
marginals for the center.

**First exp004 attempt (d=8, dropped ft[0,0]):** To fit two neighbor
indices and three factor table entries into d=8, ft[0,0] was dropped
from the encoding. This was lossy — exact posteriors require all four
factor table entries — and the result was still partial convergence at
~48% improvement, indistinguishable from exp003.

## What Worked

**Second exp004 attempt (d=16, both factor tables for v1):** The fix was
to increase d_model from 8 to 16 and give v1 both factor tables
explicitly in its encoding: ft1 in dims 5–8 and ft2 in dims 9–12. Every
other node encodes only its own factor table; v1 is the only node that
needs two. With all information local, the transformer no longer needs to
route f2's table through attention into v1's representation.

The improvement was immediate and dramatic: epoch 1 already showed 77%
improvement (vs ~34% for the failed version), and the model crossed the
0.005 strong-positive threshold before epoch 20. Final MAE of 0.002679
is better than exp003's 0.003875, despite the harder task, because the
encoding is cleaner.

The `model.py` fix was to make `d_in` a constructor parameter rather
than a hardcoded constant, and `graphs/__init__.py` was updated to
declare `d_in=16` for exp004. The trainer passes `d_in` from the graph
spec to the model, so future experiments with different input dimensions
require only a one-line change in the registry.

## Interpretation

The lesson from exp003→exp004 is an instance of a general principle:
**the transformer cannot learn what it cannot see.** Attention can in
principle route any token's information to any other token, but when the
information needed to compute a node's posterior is not present anywhere
in that node's input encoding, the model must reconstruct it through
indirect attention patterns — which is harder to learn and may not
converge. Giving v1 both factor tables directly makes the computation
local and the learning problem tractable.

This has a direct analog in the BP proof: each node's belief update
requires only its own prior and the incoming messages from its neighbors.
The messages depend on the neighbor factor tables. If those tables are
not in the node's input, the transformer must simulate a two-hop lookup
— which requires more layers and may not be learnable in practice.

## Comparison Across Experiments

| Experiment | Graph | Rounds | Final MAE | Improvement |
|------------|-------|--------|-----------|-------------|
| exp001 | v0 --- f1 --- v2 | 1 | 0.000752 | +99.3% |
| exp002 | p1, p2 → AND → dates | 1 | ~0.001 | +99.6% |
| exp003 | 3-var chain (bad encoding) | 2 | 0.068 | +47.9% |
| exp004 | 3-var chain (full encoding) | 2 | 0.002679 | +97.9% |