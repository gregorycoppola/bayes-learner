"""Health check — local only, no server."""

def add_subparser(subparsers):
    p = subparsers.add_parser("health", help="Check environment")
    p.set_defaults(func=cmd_health)

def cmd_health(args):
    import torch
    print(f"✓ bayes-learner v0.1.0")
    print(f"  torch {torch.__version__}")
    print(f"  device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")