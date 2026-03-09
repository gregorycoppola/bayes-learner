from bayes_learner.core.graphs.exp001 import make_graph as make_graph_exp001
from bayes_learner.core.graphs.exp001 import make_dataset as make_dataset_exp001
from bayes_learner.core.graphs.exp002 import make_graph as make_graph_exp002
from bayes_learner.core.graphs.exp002 import make_dataset as make_dataset_exp002
from bayes_learner.core.graphs.exp003 import make_graph as make_graph_exp003
from bayes_learner.core.graphs.exp003 import make_dataset as make_dataset_exp003
from bayes_learner.core.graphs.exp004 import make_graph as make_graph_exp004
from bayes_learner.core.graphs.exp004 import make_dataset as make_dataset_exp004

GRAPHS = {
    "exp001": {
        "make_graph": make_graph_exp001,
        "make_dataset": make_dataset_exp001,
        "description": "Two-variable symmetric factor graph (v0 --- f1 --- v2)",
        "n_rounds": 1,
        "d_in": 8,
    },
    "exp002": {
        "make_graph": make_graph_exp002,
        "make_dataset": make_dataset_exp002,
        "description": "AND/OR graph: p1, p2 -> AND -> dates (hard AND, P(dates)=p1*p2)",
        "n_rounds": 1,
        "d_in": 8,
    },
    "exp003": {
        "make_graph": make_graph_exp003,
        "make_dataset": make_dataset_exp003,
        "description": "Chain of 3 variables: v0 --- f1 --- v1 --- f2 --- v2 (2 BP rounds)",
        "n_rounds": 2,
        "d_in": 8,
    },
    "exp004": {
        "make_graph": make_graph_exp004,
        "make_dataset": make_dataset_exp004,
        "description": "Chain with explicit two-neighbor encoding for v1 (v0---f1---v1---f2---v2)",
        "n_rounds": 2,
        "d_in": 16,
    },
}

def get_graph(experiment_id: str):
    if experiment_id not in GRAPHS:
        raise ValueError(f"Unknown experiment: {experiment_id}. "
                         f"Available: {list(GRAPHS.keys())}")
    return GRAPHS[experiment_id]