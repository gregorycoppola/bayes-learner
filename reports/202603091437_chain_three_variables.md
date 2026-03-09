# exp003: Chain of Three Variables (2 BP Rounds)

**Date:** 2026-03-09  
**Experiment:** exp003  
**Status:** ✓ STRONG POSITIVE

## Setup

Graph structure: `v0 --- f1 --- v1 --- f2 --- v2`

Chain of 3 variable nodes connected by 2 factor nodes. Requires 2 rounds
of BP for exact posteriors — the first experiment testing the iterated
transformer loop.

Each round:
1. `encode(state)` — fresh token sequence, scratch slots zeroed
2. `transformer(tokens)` — one forward pass
3. `decode(output)` — read dim 0, update state beliefs

- 20,000 randomly generated graphs
- Two independent factor tables, entries drawn uniformly from [0.05, 1.0]
- Train/val split: 18,000 / 2,000
- Model: BPTransformer, d_model=32, 2 heads, 2 layers, ~26k params
- Loss: MSE on variable nodes only (v0, v1, v2 — tokens 0, 2, 4)
- Baseline MAE: 0.1310

## Results

| Metric | Value |
|--------|-------|
| Final val MAE | 0.003875 |
| Best val MAE | 0.003704 |
| Baseline MAE (predict 0.5) | 0.131042 |
| Improvement | +97.0% |
| Epochs | 50 |

## Posterior Comparison (final)

| Graph | BP exact | Transformer | Max error |
|-------|----------|-------------|-----------|
| 0 | [0.4976, 0.5505, 0.5499] | [0.5019, 0.5430, 0.5483] | 0.0075 |
| 1 | [0.6351, 0.5488, 0.4081] | [0.6372, 0.5451, 0.4052] | 0.0037 |
| 2 | [0.4827, 0.5045, 0.5301] | [0.4829, 0.5036, 0.5245] | 0.0057 |
| 3 | [0.5515, 0.3372, 0.2889] | [0.5637, 0.3392, 0.2902] | 0.0122 |
| 4 | [0.2939, 0.2760, 0.5111] | [0.2994, 0.2806, 0.5029] | 0.0082 |
| 5 | [0.4804, 0.2322, 0.2319] | [0.4824, 0.2311, 0.2323] | 0.0021 |
| 6 | [0.3748, 0.5110, 0.6334] | [0.3791, 0.5121, 0.6257] | 0.0078 |
| 7 | [0.6769, 0.6547, 0.4997] | [0.6798, 0.6535, 0.4948] | 0.0050 |
| 8 | [0.5429, 0.3508, 0.3491] | [0.5375, 0.3509, 0.3505] | 0.0054 |
| 9 | [0.4744, 0.5411, 0.5678] | [0.4774, 0.5338, 0.5633] | 0.0072 |

## Observations

Graph 3 was consistently the hardest case throughout training, with max
error starting at 0.0350 at epoch 10 and converging to 0.0122 by epoch
50. The belief pattern [0.55, 0.34, 0.29] — asymmetric beliefs pulling
away from 0.5 — requires more propagation work than symmetric cases.
It continued improving every checkpoint and did not plateau.

## Interpretation

The iterated transformer loop works. The same ~26k parameter model that
learned single-round BP in exp001 now learns two-round BP on a longer
chain, achieving 97% improvement over baseline.

This is the empirical confirmation of `transformer_iterated_implements_runBP`
from transformer-bp-lean: T transformer passes implement T rounds of BP.
The encode/decode interface between rounds — zeroing scratch slots, updating
dim 0 with new beliefs, preserving neighbor indices and factor tables — is
the right abstraction. The transformer does not need to see its own raw
output; it needs a fresh encoding of the updated belief state.

Final MAE of 0.003875 is slightly higher than exp001 (0.000752) as expected
— the two-round task is harder and the encoding for v1 (which has two
factor neighbors but only one explicitly encoded) leaves some information
implicit. Further encoding improvements or more epochs would likely close
this gap.

## Comparison Across Experiments

| Experiment | Graph | Rounds | Final MAE | Improvement |
|------------|-------|--------|-----------|-------------|
| exp001 | v0 --- f1 --- v2 | 1 | 0.000752 | +99.3% |
| exp002 | p1, p2 → AND → dates | 1 | ~0.001 | +99.6% |
| exp003 | v0 --- f1 --- v1 --- f2 --- v2 | 2 | 0.003875 | +97.0% |

All three: strong positive. The architecture scales to multi-round
inference with no changes to the model — only the training loop changes.