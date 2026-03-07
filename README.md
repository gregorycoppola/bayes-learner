# bayes-learner

Empirical counterpart to [transformer-bp-lean](../transformer-bp-lean).

`transformer-bp-lean` proves formally (Lean 4, zero sorries) that there
exist transformer weights which implement exact belief propagation on a
factor graph. The proof is constructive — it exhibits the weights explicitly.

This repo asks the empirical question: does gradient descent find them?

## The Experiment

1. Generate random tree-structured factor graphs
2. Run exact belief propagation to get ground truth posteriors
3. Train a transformer to predict those posteriors from the encoded graph state
4. Evaluate: does the transformer's output match exact posteriors on held-out graphs?

If yes: gradient descent finds weights that implement exact Bayesian inference,
just as it finds weights that simulate Turing machines (see `learner`).

If no: the gap is itself a result — it characterizes what additional inductive
bias or training signal is needed to recover Bayesian inference from gradient
descent.

## Relationship to the Trilogy

| Repo | Claim | Method |
|------|-------|--------|
| `universal-lean` | Transformer is Turing complete | Lean proof |
| `learner` | Gradient descent finds TM-simulation weights | Experiment |
| `transformer-bp-lean` | Transformer implements exact BP | Lean proof |
| `bayes-learner` | Gradient descent finds BP weights | **This repo** |

## Status

Bootstrapping. Health check only.

## Running

```bash
uv sync
bayes-learner health