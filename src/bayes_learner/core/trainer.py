"""Training loop."""
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
    init: str = "constructed",
    noise: float = 0.01,
    ffn_mode: str = "learned",
    device: str = "cpu",
) -> dict:

    print("=" * 60)
    print("BAYES-LEARNER EXPERIMENT")
    print(f"Init: {init}  Noise: {noise}  FFN: {ffn_mode}")
    print("=" * 60)

    print(f"\n[DATA] Generating {n_graphs} graphs (n_vars={n_vars})...")
    t0 = time.time()
    X, Y, var_mask = make_dataset(n_graphs, n_vars)
    print(f"[DATA] Done in {time.time()-t0:.1f}s  "
          f"X={list(X.shape)}  Y={list(Y.shape)}")

    Y_vars = Y[var_mask]
    baseline_mae = (Y_vars - 0.5).abs().mean().item()
    print(f"[DATA] Variable belief stats — "
          f"min:{Y_vars.min():.4f}  max:{Y_vars.max():.4f}  "
          f"mean:{Y_vars.mean():.4f}  std:{Y_vars.std():.4f}")
    print(f"[DATA] Baseline MAE (predict 0.5): {baseline_mae:.4f}")

    split = int(0.9 * n_graphs)
    X_train = X[:split].to(device)
    Y_train = Y[:split].to(device)
    M_train = var_mask[:split].to(device)
    X_val   = X[split:].to(device)
    Y_val   = Y[split:].to(device)
    M_val   = var_mask[split:].to(device)
    print(f"[DATA] Train: {split}  Val: {n_graphs - split}")

    print(f"\n[MODEL] Building BPTransformer...")
    model = BPTransformer(init=init, noise=noise, ffn_mode=ffn_mode).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {n_params}")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"[MODEL]   {name}: {list(p.shape)}")

    # Sanity check attention at init
    print(f"\n[SANITY] Attention routing check at init...")
    with torch.no_grad():
        x0 = X[:1]
        n_nodes = x0.shape[1]
        Q = model.Wq0(x0)
        K = model.Wk0(x0)
        scores = torch.bmm(Q, K.transpose(1, 2))[0]
        attn   = torch.softmax(scores, dim=-1)
        nb0_idx = int(x0[0, 0, 1].item())
        print(f"[SANITY] Node 0 neighbor0_idx={nb0_idx}")
        print(f"[SANITY] Scores: {[f'{v:.2f}' for v in scores[0].tolist()]}")
        print(f"[SANITY] Attn:   {[f'{v:.4f}' for v in attn[0].tolist()]}")
        print(f"[SANITY] Attn at nb0 ({nb0_idx}): {attn[0, nb0_idx].item():.4f}  "
              f"(uniform={1/n_nodes:.4f})")

    # Oracle check: what MAE does the exact BP formula achieve?
    print(f"\n[ORACLE] Testing exact BP formula (constructed FFN)...")
    with torch.no_grad():
        oracle = BPTransformer(init="constructed", noise=0.0,
                               ffn_mode="constructed").to(device)
        oracle_pred = oracle(X_val)
        oracle_mae  = (oracle_pred - Y_val)[M_val].abs().mean().item()
        print(f"[ORACLE] MAE with exact attention + exact BP formula: {oracle_mae:.6f}")
        print(f"[ORACLE] (This is the upper bound for a single-round model)")
        # Show a few oracle posteriors
        for i in range(3):
            mask = M_val[i]
            bp_vals = Y_val[i][mask].tolist()
            or_vals = oracle_pred[i][mask].tolist()
            errs    = [abs(a-b) for a,b in zip(bp_vals, or_vals)]
            print(f"[ORACLE] Graph {i}: BP={[f'{v:.3f}' for v in bp_vals]}  "
                  f"Oracle={[f'{v:.3f}' for v in or_vals]}  "
                  f"MaxErr={max(errs):.4f}")

    if ffn_mode != "constructed":
        print(f"\n[INSPECT] Weights at init:")
        inspect_weights(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    dataset   = TensorDataset(X_train, Y_train, M_train)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n[TRAIN] {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"\n{'Ep':>5}  {'Train Loss':>12}  {'Val MAE':>10}  "
          f"{'vs Baseline':>12}  {'vs Oracle':>10}  {'Time':>6}")
    print("-" * 62)

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

            if epoch == 1 and batch_idx == 0:
                print(f"[TRAIN] Ep1 batch0 — loss={loss.item():.6f}  "
                      f"pred_mean={pred[mb].mean():.4f}  "
                      f"pred_std={pred[mb].std():.4f}")

        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae  = (val_pred - Y_val)[M_val].abs().mean().item()

        improvement  = (baseline_mae - val_mae) / baseline_mae * 100
        vs_oracle    = val_mae - oracle_mae
        ep_time      = time.time() - t_ep

        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_mae"].append(val_mae)

        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_mae:>10.6f}  "
              f"{improvement:>+11.1f}%  {vs_oracle:>+10.4f}  {ep_time:>5.2f}s")

        if epoch % inspect_every == 0:
            print(f"\n[INSPECT] Epoch {epoch}:")
            if ffn_mode != "constructed":
                inspect_weights(model)
            compare_posteriors(model, X_val, Y_val, M_val, n_examples=3)
            print()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    final_mae = results["val_mae"][-1]
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    print(f"Final val MAE:    {final_mae:.6f}")
    print(f"Oracle MAE:       {oracle_mae:.6f}")
    print(f"Baseline MAE:     {baseline_mae:.6f}")
    print(f"Improvement:      {improvement:+.1f}%")
    print(f"Gap to oracle:    {final_mae - oracle_mae:+.6f}")
    if final_mae <= oracle_mae * 1.1:
        print(f"Result: ✓ STRONG POSITIVE — matches oracle (exact BP)")
    elif improvement > 30:
        print(f"Result: ~ PARTIAL — learning but not reaching oracle")
    else:
        print(f"Result: ✗ NEGATIVE — not learning")

    if ffn_mode != "constructed":
        inspect_weights(model)
    compare_posteriors(model, X_val, Y_val, M_val, n_examples=5)
    return results