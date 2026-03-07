"""Training loop for BPTransformer."""
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
) -> dict:
    print(f"Generating {n_graphs} graphs (n_vars={n_vars})...")
    X, Y = make_dataset(n_graphs, n_vars)

    split = int(0.9 * n_graphs)
    X_train, Y_train = X[:split].to(device), Y[:split].to(device)
    X_val, Y_val = X[split:].to(device), Y[split:].to(device)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BPTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\n{'Ep':>5}  {'Train Loss':>12}  {'Val MAE':>10}")
    print("-" * 34)

    results = {"epochs": [], "train_loss": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mae = (val_pred - Y_val.float()).abs().mean().item()

        results["epochs"].append(epoch)
        results["train_loss"].append(train_loss)
        results["val_mae"].append(val_mae)

        if epoch % print_every == 0:
            print(f"{epoch:>5}  {train_loss:>12.6f}  {val_mae:>10.6f}")

    print("-" * 34)
    print(f"Done. Final val MAE: {results['val_mae'][-1]:.6f}")
    return results