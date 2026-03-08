"""Training loop — logs everything useful for development."""
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from bayes_learner.core.model import BPTransformer
from bayes_learner.core.graph import make_dataset
from bayes_learner.core.inspect import inspect_weights, compare_posteriors


def train(
    n_graphs: int = 10000,
    n_vars: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    inspect_every: int = 25,
    device: str = "cpu",
) -> dict:

    print("=" * 60)
    print("BAYES-LEARNER EXPERIMENT")
    print("Does gradient descent find BP weights?")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    print(f"\n[DATA] Generating {n_graphs} graphs (n_vars={n_vars})...")
    t0 = time.time()
    X, Y, var_mask = make_dataset(n_graphs, n_vars)
    print(f"[DATA] Done in {time.time()-t0:.1f}s")
    print(f"[DATA] X shape: {list(X.shape)}  Y shape: {list(Y.shape)}")
    print(f"[DATA] Nodes per graph: {X.shape[1]} "
          f"({var_mask[0].sum().item()} var, "
          f"{(~var_mask[0]).sum().item()} factor)")

    Y_vars = Y[var_mask]
    baseline_mae = (Y_vars - 0.5).abs().mean().item()
    print(f"[DATA] Variable belief stats — "
          f"min:{Y_vars.min():.4f}  max:{Y_vars.max():.4f}  "
          f"mean:{Y_vars.mean():.4f}  std:{Y_vars.std():.4f}")
    print(f"[DATA] Baseline MAE (predict 0.5): {baseline_mae:.4f}")

    # Sample a few BP outputs
    print(f"[DATA] Sample BP posteriors (var nodes only):")
    for i in range(3):
        vals = Y[i][var_mask[i]].tolist()
        print(f"  Graph {i}: {[f'{v:.4f}' for v in vals]}")

    # ── Split ─────────────────────────────────────────────────
    split = int(0.9 * n_graphs)
    X_train = X[:split].to(device)
    Y_train = Y[:split].to(device)
    M_train = var_mask[:split].to(device)
    X_val   = X[split:].to(device)
    Y_val   = Y[split:].to(device)
    M_val   = var_mask[split:].to(device)
    print(f"[DATA] Train: {split}  Val: {n_graphs-split}")

    # ── Model ─────────────────────────────────────────────────
    print(f"\n[MODEL] Building BPTransformer...")
    model = BPTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Total parameters: {n_params}")
    for name, p in model.named_parameters():
        print(f"[MODEL]   {name}: {list(p.shape)}")

    # ── Initial weight inspection ──────────────────────────────
    print(f"\n[INSPECT] Weights at init (should be random/near-zero):")
    inspect_weights(model)

    # ── Training ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset   = TensorDataset(X_train, Y_train, M_train)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"\n[TRAIN] Starting — {epochs} epochs, "
          f"batch_size={batch_size}, lr={lr}")
    print(f"[TRAIN] Batches per epoch: {len(loader)}")
    print(f"\n{'Ep':>5}  {'Train Loss':>12}  {'Val MAE':>10}  "
          f"{'vs Baseline':>12}  {'Time':>6}")
    print("-" * 52)

    results = {"epochs": [], "train_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss, total_n = 0.0, 0

        for batch_idx, (xb, yb, mb) in enumerate(loader):
            pred = model(xb)
            diff = (pred - yb) ** 2
            loss = diff[mb].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mb.sum().item()
            total_n    += mb.sum().item()

            # Log first batch of first epoch in detail
            if epoch == 1 and batch_idx == 0:
                print(f"[TRAIN] Ep1 batch0 — loss={loss.item():.6f}  "
                      f"pred_mean={pred[mb].mean():.4f}  "
                      f"pred_std={pred[mb].std():.4f}  "
                      f"pred_min={pred[mb].min():.4f}  "
                      f"pred_max={pred[mb].max():.4f}")

        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae  = (val_pred - Y_val)[M_val].abs().mean().item()

        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_mae"].append(val_mae)

        improvement = (baseline_mae - val_mae) / baseline_mae * 100
        ep_time = time.time() - t_ep
        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_mae:>10.6f}  "
              f"{improvement:>+11.1f}%  {ep_time:>5.2f}s")

        # Periodic inspection
        if epoch % inspect_every == 0:
            print(f"\n[INSPECT] Epoch {epoch}:")
            inspect_weights(model)
            compare_posteriors(model, X_val, Y_val, M_val, n_examples=3)
            print()

    # ── Final report ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    final_mae = results["val_mae"][-1]
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    print(f"Final val MAE:    {final_mae:.6f}")
    print(f"Baseline MAE:     {baseline_mae:.6f}")
    print(f"Improvement:      {improvement:+.1f}%")
    if final_mae < 0.01:
        print(f"Result: ✓ STRONG POSITIVE — matching BP posteriors exactly")
    elif final_mae < baseline_mae * 0.5:
        print(f"Result: ~ PARTIAL — learning but not matching exactly")
    else:
        print(f"Result: ✗ NEGATIVE — not beating baseline meaningfully")

    print(f"\n[INSPECT] Final weights:")
    inspect_weights(model)
    compare_posteriors(model, X_val, Y_val, M_val, n_examples=5)

    return results