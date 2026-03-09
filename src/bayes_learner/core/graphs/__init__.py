from bayes_learner.core.graphs.exp001 import make_graph as make_graph_exp001
from bayes_learner.core.graphs.exp001 import make_dataset as make_dataset_exp001
from bayes_learner.core.graphs.exp002 import make_graph as make_graph_exp002
from bayes_learner.core.graphs.exp002 import make_dataset as make_dataset_exp002

GRAPHS = {
    "exp001": {
        "make_graph": make_graph_exp001,
        "make_dataset": make_dataset_exp001,
        "description": "Two-variable symmetric factor graph (v0 --- f1 --- v2)",
    },
    "exp002": {
        "make_graph": make_graph_exp002,
        "make_dataset": make_dataset_exp002,
        "description": "AND/OR graph: p1, p2 -> AND -> dates (hard AND, P(dates)=p1*p2)",
    },
}


def get_graph(experiment_id: str):
    if experiment_id not in GRAPHS:
        raise ValueError(f"Unknown experiment: {experiment_id}. "
                         f"Available: {list(GRAPHS.keys())}")
    return GRAPHS[experiment_id]