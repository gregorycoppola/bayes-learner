"""Training loop."""
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from bayes_learner.core.model import BPTransformer
from bayes_learner.core.graphs import get_graph


def train(
    experiment: str = "exp001",
    n_graphs: int = 20000,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 32,
    n_heads: int = 2,
    n_layers: int = 2,
    inspect_every: int = 10,
    device: str = "cpu",
) -> dict:

    graph_spec = get_graph(experiment)
    make_dataset = graph_spec["make_dataset"]
    make_graph   = graph_spec["make_graph"]
    n_rounds     = graph_spec.get("n_rounds", 1)
    d_in         = graph_spec.get("d_in", 8)

    print("=" * 60)
    print(f"BAYES-LEARNER EXPERIMENT: {experiment}")
    print(f"  {graph_spec['description']}")
    print(f"  BP rounds per inference: {n_rounds}")
    print("Can a transformer learn exact Bayesian posteriors?")
    print("=" * 60)

    print(f"\n[DATA] Generating {n_graphs} graphs...")
    t0 = time.time()
    X, Y, var_mask = make_dataset(n_graphs)
    print(f"[DATA] Done in {time.time()-t0:.1f}s")
    print(f"[DATA] X={list(X.shape)}  Y={list(Y.shape)}")

    Y_vars = Y[var_mask]
    baseline_mae = (Y_vars - 0.5).abs().mean().item()
    print(f"[DATA] Posterior stats — "
          f"min:{Y_vars.min():.4f}  max:{Y_vars.max():.4f}  "
          f"mean:{Y_vars.mean():.4f}  std:{Y_vars.std():.4f}")
    print(f"[DATA] Baseline MAE (always predict 0.5): {baseline_mae:.4f}")

    print(f"[DATA] Sample posteriors:")
    for i in range(5):
        g = make_graph()
        print(f"  {g.exact_posteriors()}")

    split = int(0.9 * n_graphs)
    X_train, Y_train, M_train = X[:split], Y[:split], var_mask[:split]
    X_val,   Y_val,   M_val   = X[split:], Y[split:], var_mask[split:]
    print(f"[DATA] Train: {split}  Val: {n_graphs-split}")

    print(f"\n[MODEL] BPTransformer — "
          f"d_in={d_in}, d_model={d_model}, heads={n_heads}, layers={n_layers}")
    model = BPTransformer(d_in=d_in, d_model=d_model,
                          n_heads=n_heads, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)
    dataset = TensorDataset(X_train, Y_train, M_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n[TRAIN] {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"\n{'Ep':>5}  {'Loss':>10}  {'Val MAE':>10}  "
          f"{'Baseline':>10}  {'Improv':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 64)

    results = {"epochs": [], "train_loss": [], "val_mae": []}
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss, total_n = 0.0, 0

        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pred = _forward_n_rounds(model, xb, n_rounds, d_in)
            loss = ((pred - yb) ** 2)[mb].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mb.sum().item()
            total_n    += mb.sum().item()

        train_loss = total_loss / total_n
        model.eval()
        with torch.no_grad():
            val_pred = _forward_n_rounds(model, X_val.to(device), n_rounds, d_in)
            val_mae  = (val_pred - Y_val.to(device))[M_val.to(device)].abs().mean().item()

        scheduler.step(val_mae)
        improvement = (baseline_mae - val_mae) / baseline_mae * 100
        current_lr  = optimizer.param_groups[0]["lr"]
        ep_time     = time.time() - t_ep

        if val_mae < best_mae:
            best_mae = val_mae

        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_mae"].append(val_mae)

        print(f"{epoch:>5}  {train_loss:>10.6f}  {val_mae:>10.6f}  "
              f"{baseline_mae:>10.6f}  {improvement:>+7.1f}%  "
              f"{current_lr:>8.6f}  {ep_time:>5.2f}s")

        if epoch % inspect_every == 0:
            _compare_posteriors(model, X_val, Y_val, M_val, device,
                                n_rounds=n_rounds, d_in=d_in, n=5)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    final_mae   = results["val_mae"][-1]
    improvement = (baseline_mae - final_mae) / baseline_mae * 100
    print(f"Final val MAE:  {final_mae:.6f}")
    print(f"Best val MAE:   {best_mae:.6f}")
    print(f"Baseline MAE:   {baseline_mae:.6f}")
    print(f"Improvement:    {improvement:+.1f}%")
    if final_mae < 0.005:
        print("Result: ✓ STRONG POSITIVE — near-exact posterior matching")
    elif improvement > 50:
        print("Result: ~ GOOD — substantial improvement over baseline")
    elif improvement > 20:
        print("Result: ~ PARTIAL — learning but not converged")
    else:
        print("Result: ✗ NEGATIVE — not learning")

    _compare_posteriors(model, X_val, Y_val, M_val, device,
                        n_rounds=n_rounds, d_in=d_in, n=10)
    return results


def _forward_n_rounds(model, x: torch.Tensor, n_rounds: int,
                      d_in: int = 8) -> torch.Tensor:
    """
    Run the transformer for n_rounds, updating belief (dim 0) between rounds.
    Only dim 0 (belief) is updated between rounds — all other dims are static
    graph structure and factor tables, which should not be zeroed.
    """
    out = x
    for _ in range(n_rounds):
        pred = model(out)
        out = out.clone()
        out[:, :, 0] = pred   # update beliefs only
    return pred


def _compare_posteriors(model, X_val, Y_val, M_val, device,
                        n_rounds: int = 1, d_in: int = 8, n: int = 5):
    model.eval()
    print(f"\n{'─'*60}")
    print(f"POSTERIOR COMPARISON (BP exact vs Transformer, {n_rounds} round(s))")
    print(f"{'─'*60}")
    with torch.no_grad():
        pred = _forward_n_rounds(model, X_val[:n].to(device), n_rounds, d_in)
    for i in range(n):
        mask = M_val[i]
        bp   = Y_val[i][mask].tolist()
        tf   = pred[i][mask].tolist()
        errs = [abs(a - b) for a, b in zip(bp, tf)]
        print(f"Graph {i:2d}:  "
              f"BP=[{', '.join(f'{v:.4f}' for v in bp)}]  "
              f"TF=[{', '.join(f'{v:.4f}' for v in tf)}]  "
              f"MaxErr={max(errs):.4f}  MeanErr={sum(errs)/len(errs):.4f}")
    print(f"{'─'*60}")