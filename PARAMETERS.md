# Parameters

All parameters for the `bayes-learner train` command.

---

## Data Parameters

### `--graphs` (default: 10000)
Number of factor graphs to generate. 90% used for training, 10% for validation.

The earlier run used `--graphs 100` which gave only 90 training graphs —
almost certainly too few. The model has ~700 parameters and 90 graphs
with 5 variable nodes each = 450 training examples. Severe underfitting risk.
**Recommendation: at least 5000, ideally 10000.**

### `--vars` (default: 5)
Number of variable nodes per graph. Total nodes = `2*vars - 1`
(variable nodes at even indices, factor nodes at odd indices, chain topology).

With `--vars 5`: 9 nodes per graph (5 variable, 4 factor).
The chain has diameter 8, meaning exact BP requires 8 message-passing rounds
to propagate information from one end to the other.
A single transformer forward pass can only do 1 round.
**This means the single-pass model cannot learn exact posteriors for
nodes far from each other — it is structurally limited to local corrections.**

---

## Training Parameters

### `--epochs` (default: 100)
Number of full passes through the training data.

### `--batch-size` (default: 64)
Number of graphs per gradient update. With 9000 training graphs,
batch=64 gives ~140 batches per epoch.

The earlier run used `--graphs 100` which gave only 1 batch per epoch
(all 90 training graphs fit in one batch of 64... barely). This means
the gradient estimate was based on very few examples per update.

### `--lr` (default: 1e-3)
Adam learning rate.

---

## Model Parameters

### `--init` (default: constructed)
How to initialize the 6 attention weight matrices (Wq0, Wk0, Wv0, Wq1, Wk1, Wv1).

- `constructed` — initialize from the explicit weights proven in `Attention.lean`,
  plus Gaussian noise of std `--noise`. This gives the attention heads a strong
  starting point that already implements correct neighbor routing.
- `random` — standard Kaiming initialization. Expected to fail because random
  weights produce uniform softmax attention (all neighbors weighted equally),
  which gives near-zero gradient signal.

### `--noise` (default: 0.01)
Standard deviation of Gaussian noise added to constructed initialization.
`0.0` = exact construction (no noise). `0.1` = substantial perturbation.
Controls how far from the construction we start and whether gradient descent
finds its way back.

### `--ffn` (default: learned)
How the FFN computes the belief update after attention.

- `learned` — a 3-input MLP (`[belief, msg0, msg1] → updated_belief`).
  Tests whether gradient descent finds the BP update formula.
- `constructed` — the exact BP update formula with no learned parameters:
  `sigmoid(logit(belief) + logit(msg0) + logit(msg1))`
  This is the oracle — it tells us the ceiling MAE for a single-round model,
  and also verifies the attention routing is correct independent of the FFN.

---

## Diagnostic Parameters

### `--inspect-every` (default: 25)
Print full weight matrices and posterior comparisons every N epochs.

---

## Key Experiments

| Command | What it tests |
|---------|--------------|
| `--ffn constructed --epochs 1` | Oracle ceiling: how good is 1 round of exact BP? |
| `--init constructed --ffn learned` | **Main experiment**: does SGD find the BP update? |
| `--init random --ffn learned` | Baseline failure: confirms random init doesn't work |
| `--init constructed --noise 0.0 --ffn learned` | Does exact init help more than noisy? |
| `--vars 3` | Smaller graphs (diameter 4) — easier for single-pass model |

---

## Known Issues

**`--graphs 100` is too small.** The earlier debug run used 100 graphs.
The attention weights converged correctly (match scores ~0.7-1.2) but
the FFN learned a constant function. With only 90 training graphs the
FFN has almost no signal — the model can minimize loss by ignoring the
FFN input and outputting a constant near the mean belief (~0.5).

**Single-pass limitation.** The chain topology with `--vars 5` has
diameter 8. One transformer forward pass = one round of BP = only
immediate neighbors influence each node's belief. Nodes in the middle
of the chain need information from both ends, which requires multiple
rounds. The oracle MAE (`--ffn constructed`) will show exactly how
large this limitation is.