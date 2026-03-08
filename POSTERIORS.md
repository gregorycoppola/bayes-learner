# Posteriors

## What a Posterior Is

In probability theory, a **posterior** is a probability distribution over
some unknown quantity, updated in light of evidence.

The starting point is a **prior** — your belief about something before
seeing any data. For example: "this coin is probably fair, so P(heads) = 0.5."

When you observe evidence, you update using Bayes' rule:

    P(hypothesis | evidence) ∝ P(evidence | hypothesis) * P(hypothesis)

The result is the **posterior** — your updated belief given what you observed.
It encodes exactly how much the evidence should shift your prior.

The posterior is the rational answer to the question: given everything I know,
what should I believe?

---

## Posteriors in a Bayesian Network

A **Bayesian network** (or factor graph) is a structured model where:

- There are several unknown variables (e.g. X1, X2, X3)
- Each variable has a prior belief (uniform = 0.5 for binary variables)
- **Factor nodes** encode relationships between variables — e.g.
  "if X1=1 then X2 is likely 1 too", represented as a table of joint
  probabilities f(X1, X2)

The **marginal posterior** for variable Xi is:

    P(Xi = 1 | all factors) ∝ Σ_{all other variables} Π_{factors} f(...)

This is the probability that Xi takes value 1, after integrating out
all other variables and accounting for all factor constraints.

Computing this exactly requires summing over all combinations of all
other variables — exponential in the number of variables. **Belief
propagation (BP)** is an algorithm that computes exact marginals
efficiently on tree-structured graphs by passing messages between nodes.

The output of BP is a vector of marginal posteriors — one float per
variable node, each in [0, 1] — representing the rational belief about
each variable given the full factor graph structure.

---

## What We Expect to Happen Here

We generate random factor graphs. Each graph has variable nodes connected
by factor nodes encoding random pairwise relationships. We run exact BP
on each graph to get the ground truth posterior vector.

Then we train a transformer on those posteriors.

### What the Transformer Sees

Input: the encoded factor graph state — for each node, its type,
its neighbor indices, and its factor table. Initial beliefs are all 0.5
(uniform prior, no information yet).

Target: the exact BP posterior for each variable node.

### What We Want to See

After training on enough graphs, the transformer should output a posterior
vector that closely matches the exact BP posteriors on held-out graphs.

Concretely, for a held-out graph:

    BP posterior:          [0.823, 0.612, 0.441]
    Transformer posterior: [0.819, 0.608, 0.447]
    Max error: 0.008

The transformer has learned to do Bayesian reasoning — to combine the
factor table constraints and propagate beliefs across the graph structure,
arriving at the same answer that exact inference would give.

### Why This Would Be Significant

Current transformers (GPT, Claude, etc.) produce outputs that are not
grounded in any principled probabilistic model. They can assert things
confidently that are not supported by their training data, or assign
equal confidence to contradictory claims. This is the hallucination problem.

A transformer that has learned to compute exact posteriors over a
structured knowledge base would be different in kind. Its outputs would
be **calibrated** — confidence proportional to actual evidential support.
It couldn't hallucinate in the strong sense because its outputs would
be constrained to be consistent with the factor graph structure.

The formal result in `transformer-bp-lean` says such weights *exist*.
This experiment asks whether a transformer can *learn* to behave this way
from data — without being hand-coded, just by training on examples of
correct Bayesian inference.

If it can, the path to a no-hallucination transformer becomes:

1. Encode your knowledge base as a factor graph (or QBBN)
2. Train a transformer on that knowledge base using exact BP as supervision
3. The trained transformer computes calibrated posteriors at inference time

### What We Are Not Claiming (Yet)

We are not claiming that existing large language models already do this.
We are not claiming that the specific weights learned will match the
constructed weights from `Attention.lean`.

We are asking a narrower question: **can a transformer learn to compute
exact Bayesian posteriors on factor graphs at all?** If the answer is yes,
even on small toy graphs, that is the first empirical step toward the
broader claim.