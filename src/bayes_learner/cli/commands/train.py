"""Train command — runs locally with per-epoch streaming output."""
from bayes_learner.core.trainer import train

def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train transformer on BP inference")
    p.add_argument("--graphs", type=int, default=5000)
    p.add_argument("--vars", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.set_defaults(func=cmd_train)

def cmd_train(args):
    train(
        n_graphs=args.graphs,
        n_vars=args.vars,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        print_every=1,
    )