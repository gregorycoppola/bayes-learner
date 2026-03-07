"""Train command."""
import json
from bayes_learner.cli.client import get_client

def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train transformer on BP inference")
    p.add_argument("--url", default="http://localhost:8001")
    p.add_argument("--graphs", type=int, default=5000)
    p.add_argument("--vars", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.set_defaults(func=cmd_train)

def cmd_train(args):
    payload = {
        "n_graphs": args.graphs,
        "n_vars": args.vars,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    print(f"Training: {args.graphs} graphs, {args.vars} vars, {args.epochs} epochs...")
    try:
        with get_client(args.url) as client:
            resp = client.post("/api/train", json=payload, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            print(f"✓ Done — final val MAE: {data['final_val_mae']:.6f}")
    except Exception as e:
        print(f"✗ {e}")
        raise SystemExit(1)