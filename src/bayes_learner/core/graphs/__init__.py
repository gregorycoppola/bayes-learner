from bayes_learner.core.graphs.exp001 import make_graph, make_dataset

GRAPHS = {
    "exp001": {
        "make_graph": make_graph,
        "make_dataset": make_dataset,
        "description": "Two-variable symmetric factor graph",
    },
}

def get_graph(experiment_id: str):
    if experiment_id not in GRAPHS:
        raise ValueError(f"Unknown experiment: {experiment_id}. "
                         f"Available: {list(GRAPHS.keys())}")
    return GRAPHS[experiment_id]