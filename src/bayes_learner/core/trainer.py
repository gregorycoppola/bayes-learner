"""Training loop for BPTransformer."""
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from bayes_learner.core.model import BPTransformer
from bayes_learner.core.graph import make_dataset


def train(
    n_graphs: int = 5000,
    n_vars: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    print_every: int = 1,
    analyze: bool = False,
) -> dict:
    print(f"[1/6] Generating {n_graphs} graphs (n_vars={n_vars})...")
    t0 = time.time()
    X, Y = make_dataset(n_graphs, n_vars)
    print(f"      Done in {time.time()-t0:.1f}s  X={list(X.shape)}  Y={list(Y.shape)}")

    print(f"[2/6] Analyzing targets...")
    var_mask = (X[:, :, 3] == 0)  # [n_graphs, n], True = variable node
    fac_mask = ~var_mask
    Y_vars = Y[var_mask]
    Y_facs = Y[fac_mask]
    n_nodes = X.shape[1]
    n_var_nodes = var_mask[0].sum().item()
    n_fac_nodes = fac_mask[0].sum().item()
    print(f"      Nodes per graph: {n_nodes} ({n_var_nodes} variable, {n_fac_nodes} factor)")
    print(f"      Variable beliefs — min:{Y_vars.min():.4f}  max:{Y_vars.max():.4f}  "
          f"mean:{Y_vars.mean():.4f}  std:{Y_vars.std():.4f}")
    print(f"      Factor beliefs   — min:{Y_facs.min():.4f}  max:{Y_facs.max():.4f}  "
          f"mean:{Y_facs.mean():.4f}  std:{Y_facs.std():.4f}")
    print(f"      Fraction var beliefs near 0.5 (|b-0.5|<0.05): "
          f"{((Y_vars - 0.5).abs() < 0.05).float().mean():.3f}")
    print(f"      Baseline MAE (predict 0.5 always): {(Y_vars - 0.5).abs().mean():.4f}")

    if analyze:
        from bayes_learner.core.graph import make_tree, run_bp
        print(f"\n      Sample graphs:")
        for i in range(5):
            g = make_tree(n_vars)
            bp = run_bp(g)
            var_beliefs = [f"{bp[j]:.3f}" for j in range(g.n) if g.node_type[j] == 0]
            print(f"        Graph {i}: var beliefs = {var_beliefs}")
        return {}

    print(f"[3/6] Splitting data (90/10 train/val)...")
    split = int(0.9 * n_graphs)
    X_train, Y_train = X[:split].to(device), Y[:split].to(device)
    X_val,   Y_val   = X[split:].to(device), Y[split:].to(device)
    M_train = var_mask[:split].to(device)
    M_val   = var_mask[split:].to(device)
    print(f"      Train: {split} graphs  Val: {n_graphs-split} graphs")
    print(f"      Train var nodes: {M_train.sum().item()}  Val var nodes: {M_val.sum().item()}")

    print(f"[4/6] Building model...")
    model = BPTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      BPTransformer — {n_params} parameters")
    for name, p in model.named_parameters():
        print(f"        {name}: {list(p.shape)}")

    print(f"[5/6] Setting up optimizer (Adam, lr={lr})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, Y_train, M_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(loader)
    print(f"      Batch size: {batch_size}  Batches per epoch: {n_batches}")

    print(f"[6/6] Training for {epochs} epochs...")
    print(f"\n{'Ep':>5}  {'Train Loss':>12}  {'Val MAE (var)':>14}  {'Time':>6}")
    print("-" * 46)

    results = {"epochs": [], "train_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch_idx, (xb, yb, mb) in enumerate(loader):
            pred = model(xb)
            diff = (pred - yb.float()) ** 2
            loss = diff[mb].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mb.sum().item()
            total_n += mb.sum().item()
            if epoch == 1 and batch_idx == 0:
                print(f"      First batch — loss={loss.item():.6f}  "
                      f"pred_mean={pred[mb].mean():.4f}  pred_std={pred[mb].std():.4f}")
        train_loss = total_loss / total_n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = (val_pred - Y_val.float())[M_val].abs().mean().item()
            if epoch == 1:
                print(f"      Epoch 1 val — pred_mean={val_pred[M_val].mean():.4f}  "
                      f"pred_std={val_pred[M_val].std():.4f}  "
                      f"target_mean={Y_val[M_val].mean():.4f}")

        ep_time = time.time() - t_ep
        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_mae"].append(val_mae)

        if epoch % print_every == 0:
            print(f"{epoch:>5}  {train_loss:>12.6f}  {val_mae:>14.6f}  {ep_time:>5.1f}s")

    print("-" * 46)
    baseline = (Y_vars - 0.5).abs().mean().item()
    final_mae = results['val_mae'][-1]
    print(f"Done. Final val MAE: {final_mae:.6f}  Baseline (0.5): {baseline:.6f}  "
          f"{'✓ beating baseline' if final_mae < baseline else '✗ not beating baseline'}")
    return results