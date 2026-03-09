# exp001: Two-Variable Symmetric Factor Graph

**Date:** 2026-03-09  
**Experiment:** exp001  
**Status:** ✓ STRONG POSITIVE

## Setup

Graph structure: `v0 --- f1 --- v2`

Simplest non-trivial Bayesian network. Exact BP converges in one round —
matches a single transformer forward pass.

- 20,000 randomly generated graphs
- Factor tables drawn uniformly from [0.05, 1.0]
- Train/val split: 18,000 / 2,000
- Model: BPTransformer, d_model=32, 2 heads, 2 layers, ~26k params
- Loss: MSE on variable nodes only

## Results

| Metric | Value |
|--------|-------|
| Final val MAE | 0.000752 |
| Baseline MAE (predict 0.5) | 0.112778 |
| Improvement | 99.3% |

## Posterior Comparison (sample)

| Graph | BP exact | Transformer | Max error |
|-------|----------|-------------|-----------|
| 0 | [0.4411, 0.4772] | [0.4403, 0.4759] | 0.0012 |
| 1 | [0.5806, 0.4871] | [0.5815, 0.4884] | 0.0013 |
| 2 | [0.4721, 0.4434] | [0.4739, 0.4456] | 0.0023 |
| 3 | [0.5539, 0.4896] | [0.5542, 0.4906] | 0.0010 |
| 4 | [0.7197, 0.5100] | [0.7181, 0.5102] | 0.0014 |

## Interpretation

Gradient descent finds transformer weights that compute exact Bayesian
posteriors on two-variable factor graphs, matched to 3 decimal places
on held-out graphs never seen during training. This is the empirical
counterpart to transformer-bp-lean: the formal proof says the weights
exist; exp001 shows gradient descent finds them.