"""
Weight inspection: compare learned weights to Attention.lean construction.

For each weight matrix, prints:
  - Full 8x8 matrix (rounded)
  - Argmax entry (row, col, value)
  - Expected entry from construction (row, col, value=1.0)
  - Match score: |learned[target_row][target_col] - 1.0| + sum|other entries|
    (0.0 = perfect match, higher = further from construction)
"""
import torch
from bayes_learner.core.model import BPTransformer, CONSTRUCTED


def inspect_weights(model: BPTransformer):
    print("\n" + "=" * 60)
    print("WEIGHT INSPECTION vs Attention.lean construction")
    print("=" * 60)

    scores = {}
    for name, (constructor, (exp_row, exp_col)) in CONSTRUCTED.items():
        W = getattr(model, name).weight.detach()  # [8, 8]
        W_constructed = constructor()

        # Argmax of learned weight
        flat_idx = W.abs().argmax().item()
        learned_row = flat_idx // 8
        learned_col = flat_idx % 8
        learned_val = W[learned_row, learned_col].item()

        # Value at expected position
        val_at_target = W[exp_row, exp_col].item()

        # Match score
        score = abs(val_at_target - 1.0) + (W - W_constructed).abs().sum().item()
        scores[name] = score

        # Row/col match
        row_match = "✓" if learned_row == exp_row else "✗"
        col_match = "✓" if learned_col == exp_col else "✗"

        print(f"\n{name}:")
        print(f"  Expected argmax:  row={exp_row}, col={exp_col}, val=1.0")
        print(f"  Learned argmax:   row={learned_row} {row_match}, "
              f"col={learned_col} {col_match}, val={learned_val:.4f}")
        print(f"  Val at target pos: {val_at_target:.4f}")
        print(f"  Match score: {score:.4f}  (0.0=perfect)")
        print(f"  Full matrix (learned):")
        for i, row in enumerate(W.tolist()):
            marker = " ←" if i == exp_row else ""
            print(f"    row {i}: {[f'{v:+.2f}' for v in row]}{marker}")

    print(f"\n{'─'*60}")
    print(f"Overall match scores:")
    for name, score in scores.items():
        bar = "█" * min(int(score * 5), 40)
        print(f"  {name}: {score:.4f}  {bar}")
    total = sum(scores.values())
    print(f"  Total: {total:.4f}  (0.0 = all weights match construction exactly)")
    print("=" * 60)


def compare_posteriors(model: BPTransformer, X: torch.Tensor,
                       Y: torch.Tensor, var_mask: torch.Tensor,
                       n_examples: int = 5):
    """
    Print BP posterior vs transformer posterior for n_examples graphs.
    """
    model.eval()
    print("\n" + "=" * 60)
    print("POSTERIOR COMPARISON: BP vs Transformer")
    print("=" * 60)
    with torch.no_grad():
        pred = model(X[:n_examples])  # [n_examples, n]
    for i in range(n_examples):
        mask = var_mask[i]
        bp_vals   = Y[i][mask].tolist()
        tf_vals   = pred[i][mask].tolist()
        errors    = [abs(b - t) for b, t in zip(bp_vals, tf_vals)]
        max_err   = max(errors)
        mean_err  = sum(errors) / len(errors)
        print(f"\nGraph {i}:")
        print(f"  BP posterior:          {[f'{v:.4f}' for v in bp_vals]}")
        print(f"  Transformer posterior: {[f'{v:.4f}' for v in tf_vals]}")
        print(f"  Errors:                {[f'{v:.4f}' for v in errors]}")
        print(f"  Max error: {max_err:.4f}   Mean error: {mean_err:.4f}")
    print("=" * 60)