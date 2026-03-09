"""Train command."""
from bayes_learner.core.trainer import train
from bayes_learner.core.graphs import GRAPHS


def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train transformer on exact BP posteriors")
    p.add_argument("experiment",      type=str,   nargs="?", default="exp001",
                   choices=list(GRAPHS.keys()),
                   help="Which experiment to run (default: exp001)")
    p.add_argument("--graphs",        type=int,   default=20000)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--d-model",       type=int,   default=32)
    p.add_argument("--n-heads",       type=int,   default=2)
    p.add_argument("--n-layers",      type=int,   default=2)
    p.add_argument("--inspect-every", type=int,   default=10)
    p.set_defaults(func=cmd_train)


def cmd_train(args):
    train(
        experiment=args.experiment,
        n_graphs=args.graphs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        inspect_every=args.inspect_every,
    )