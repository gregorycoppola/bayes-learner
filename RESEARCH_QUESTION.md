# Research Question

## Statement

> Does gradient descent find transformer weights that implement
> exact Bayesian inference on factor graphs?

## Background

`transformer-bp-lean` proves formally that such weights *exist*.
The proof is constructive: it exhibits explicit weight matrices
(`Wq`, `Wk`, `Wv`) and verifies in Lean 4 that one transformer
forward pass on an encoded factor graph state computes exactly
one round of belief propagation. T forward passes compute T rounds.
On tree-structured graphs, this gives exact posterior beliefs with
no approximation.

The formal result is an existence proof. It does not say gradient
descent will find these weights, or any functionally equivalent weights.

## The Experiment

Train a transformer supervised on exact BP posteriors generated from
random tree-structured factor graphs. Evaluate whether the trained
transformer's output beliefs match the exact posteriors on held-out graphs.

**Data:** randomly generated tree-structured factor graphs  
**Targets:** exact posterior beliefs, computed by running BP to convergence  
**Model:** transformer with architecture matching `transformer-bp-lean` construction  
**Metric:** posterior error per variable node; primary result is whether
this reaches zero (or near-zero) on all node types

## The Two Outcomes

**If the transformer learns exact posteriors:**
Gradient descent finds weights that implement Bayesian inference,
whether or not those weights match the explicit construction in
`Attention.lean`. This is the empirical counterpart to the Lean proof —
the same relationship `learner` has to `universal-lean`.

**If the transformer fails to learn exact posteriors:**
The gap between the formal construction and learned behavior is itself
a result. It characterizes what additional inductive bias, architecture
constraint, or training signal is needed to recover Bayesian inference
from gradient descent alone.

## Connection to the No-Hallucination Claim

The trilogy's central applied claim is that a transformer operating
over a structured knowledge base can compute exact Bayesian posterior
beliefs — and that this provides a formal no-hallucination guarantee.

This experiment tests whether that guarantee is achievable not just
in principle (the Lean proof) but in practice (via training). A positive
result means the guarantee is learnable, not just constructible.

## What We Are Not Testing

We are not primarily testing whether the *learned weights match the
constructed weights* from `Attention.lean`. Weight-level matching is
a secondary question. The primary question is functional: does the
transformer compute the right posteriors?

Weight inspection is a follow-on analysis if the functional result
is positive.