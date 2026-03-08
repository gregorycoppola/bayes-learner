# Experiment Plan

## The Question

Does gradient descent find transformer weights that implement exact
Bayesian inference on factor graphs?

The Lean proof in `transformer-bp-lean` says such weights *exist* and
exhibits them explicitly. This experiment asks whether gradient descent
finds them — or something functionally equivalent.

---

## The Constructed Weights (Ground Truth from Lean Proof)

From `Attention.lean`, the constructed weights are sparse and interpretable.

**Head 0** — gathers neighbor 0's belief into scratch slot 4:

    Wq0 = Wk0 = projectDim(1):
      row 1, col 1 = 1.0, all else 0.0
      → Q·K = emb[1] * emb[1] = (neighbor 0 index)²
      → attention concentrates on the node whose index matches neighbor 0

    Wv0 = crossProject(0 → 4):
      row 4, col 0 = 1.0, all else 0.0
      → reads dim 0 (belief), writes to dim 4 (scratch slot 0)

**Head 1** — gathers neighbor 1's belief into scratch slot 5:

    Wq1 = Wk1 = projectDim(2):
      row 2, col 2 = 1.0, all else 0.0

    Wv1 = crossProject(0 → 5):
      row 5, col 0 = 1.0, all else 0.0
      → reads dim 0 (belief), writes to dim 5 (scratch slot 1)

**FFN** — updates belief from scratch slots:

    reads dims 4 and 5 (neighbor beliefs gathered by attention)
    writes updated belief to dim 0
    implements: new_belief ∝ belief * msg_from_nb0 * msg_from_nb1

These are the targets. After training we print the learned weights
and compare them to these constructions.

---

## The Setup

**Data generation:**
- 10,000 random tree-structured factor graphs
- 5 variable nodes, 4 factor nodes per graph (chain topology)
- Random factor tables with positive entries (uniform [0.1, 1.0])
- Exact BP run to convergence → ground truth posterior per variable node
- Train/val split: 9,000 / 1,000

**Model:**
- 2-head transformer, d_model=8, matching `Attention.lean` architecture
- Separate Wq/Wk/Wv per head (not fused), no LayerNorm
- FFN: Linear(8, 32) → ReLU → Linear(32, 1) → sigmoid
- ~400 parameters total

**Training:**
- MSE loss on variable node posteriors only (factor nodes masked out)
- Adam, lr=1e-3, batch size=64
- 100 epochs

**Evaluation:**
1. Val MAE vs baseline (predict 0.5)
2. Per-graph posterior comparison: BP vector vs transformer vector
3. Weight inspection: learned Wq0, Wk0, Wv0, Wq1, Wk1, Wv1 vs constructed

---

## Expected Results and What Each Means

### Result A: Weights Match Construction (Strong Positive)

    Wv0 learned:                  Wv0 constructed:
    row 4, col 0 ≈ 1.0            row 4, col 0 = 1.0
    all other entries ≈ 0.0       all other entries = 0.0
    
    Wq0 learned:                  Wq0 constructed:
    row 1, col 1 ≈ 1.0            row 1, col 1 = 1.0
    all other entries ≈ 0.0       all other entries = 0.0

    Val MAE ≈ 0.005
    Per-graph max error < 0.01

**What it means:** Gradient descent not only finds weights that compute
the right posteriors — it finds the *same* weights the Lean proof
constructs. The formal construction is not just an existence proof;
it reflects the natural solution that optimization discovers.

This is the strongest possible result. It says the Lean proof describes
*how transformers actually work*, not just *how they could work*.

---

### Result B: Outputs Match, Weights Don't (Functional Positive)

    Val MAE ≈ 0.005  (posteriors match)

    Wv0 learned:
    row 4, col 0 ≈ 0.7   ← dominant but not clean
    row 4, col 3 ≈ 0.3   ← some noise
    other entries small

**What it means:** Gradient descent finds weights that implement
Bayesian inference functionally — the output posteriors are correct —
but via a different circuit than the constructed one. The transformer
*is* a Bayesian reasoner but took a different route to get there.

This is still a strong positive result. The no-hallucination guarantee
holds empirically. It also opens a question: what is the learned circuit
doing, and is it more or less efficient than the construction?

---

### Result C: Outputs Partially Match, Weights Noisy (Weak Positive)

    Val MAE ≈ 0.04  (beating baseline of 0.126 but not matching BP)

    Posterior comparison:
    BP:          [0.823, 0.612, 0.441, 0.756, 0.389]
    Transformer: [0.771, 0.584, 0.490, 0.701, 0.421]
    Max error: 0.052  ← correlated but off

**What it means:** The transformer has learned something Bayesian —
it's tracking the factor structure — but one forward pass is not enough.
The tree has depth 4 and messages need 4 rounds to propagate from
leaves to root. A single forward pass can only do one round of BP.

Fix: run T forward passes (agent loop). This is what the theorem
actually says — T passes implement T rounds of BP, and a depth-D tree
needs D rounds for exact inference.

---

### Result D: Not Beating Baseline (Negative)

    Val MAE ≈ 0.12  (near baseline of 0.126)
    Weights: no interpretable structure

**What it means:** The model is not learning BP at all. Either the
architecture doesn't match the construction closely enough, or the
training signal is insufficient, or there's a bug in the encoding.

Diagnosis path:
1. Check encoding — are dim assignments matching `encodeBPState`?
2. Check BP targets — are the ground truth posteriors correct?
3. Try initializing weights close to the construction and see if
   gradient descent stays there (stability test)
4. Try the agent loop (T passes) before concluding single-pass fails

---

## The Weight Inspection Protocol

After training, for each of Wq0, Wk0, Wv0, Wq1, Wk1, Wv1:

1. Print the full 8x8 matrix rounded to 2 decimal places
2. Find the argmax entry (row, col, value)
3. Compare to the constructed target (expected row, col, value)
4. Score: |learned[target_row][target_col] - 1.0| + sum of |other entries|

A perfect match scores 0.0. A random matrix scores ~8.0.

This gives a single number per weight matrix that measures how close
gradient descent got to the Lean construction.

---

## Connection to Paper 4

This experiment is Contribution 4 of Paper 4
(`transformer-bp-lean/matrix/PAPER4.md`):

> The natural next experiment: train a transformer on QBBN factor
> graph inference tasks and check whether the learned circuit matches
> the constructed circuit in Attention.lean.

A Result A or B here closes the empirical gap for Contribution 2
(the BP implementation theorem) — the same way `learner` closed the
empirical gap for Contribution 1 (Turing completeness).