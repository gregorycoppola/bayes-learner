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
) -> dict:
    """
    Train BPTransformer on random factor graphs.
    Returns a results dict with loss curve and final accuracy.
    """
    print(f"Generating {n_graphs} graphs (n_vars={n_vars})...")
    X, Y = make_dataset(n_graphs, n_vars)
    n = X.shape[1]

    # Train/val split
    split = int(0.9 * n_graphs)
    X_train, Y_train = X[:split].to(device), Y[:split].to(device)
    X_val, Y_val = X[split:].to(device), Y[split:].to(device)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BPTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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

        if epoch % 5 == 0 or epoch == 1:
            print(f"Ep {epoch:4d}  loss={train_loss:.6f}  val_mae={val_mae:.6f}")

    return results