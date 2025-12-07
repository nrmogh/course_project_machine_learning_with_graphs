DATASET_REGISTRY = {}

def register_dataset(name: str):
    def register_dataset_func(func):
        DATASET_REGISTRY[name] = func()
    return register_dataset_func

# Expose randomization utilities
from .graph_randomization import (
    randomize_graph,
    shuffle_edges_random,
    shuffle_edges_degree_preserving,
    shuffle_node_features,
    shuffle_edge_features,
    RandomizedDatasetWrapper,
)

from .randomized_dataset import (
    RandomizedPYGDataset,
    RandomizedPYGDatasetSubset,
    create_randomized_zinc_splits,
)

# SBM synthetic dataset
from .sbm_dataset import (
    generate_sbm_graph,
    SBMDataset,
    SBMDatasetSimple,
    create_sbm_splits,
    compute_p_q_from_homophily_and_degree,
)

# Backwards compatibility alias
GraphormerRandomizedPYGDataset = RandomizedPYGDatasets