"""Train command."""
from bayes_learner.core.trainer import train

def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Run the BP weight learning experiment")
    p.add_argument("--graphs",        type=int,   default=10000)
    p.add_argument("--vars",          type=int,   default=5)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--inspect-every", type=int,   default=25)
    p.add_argument("--init",          type=str,   default="constructed",
                   choices=["constructed", "random"],
                   help="constructed=start from Attention.lean weights, random=kaiming")
    p.add_argument("--noise",         type=float, default=0.01,
                   help="Noise added to constructed init")
    p.set_defaults(func=cmd_train)

def cmd_train(args):
    train(
        n_graphs=args.graphs,
        n_vars=args.vars,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        inspect_every=args.inspect_every,
        init=args.init,
        noise=args.noise,
    )