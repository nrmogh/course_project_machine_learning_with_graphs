# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train", subset=True)
            valid_set = MyZINC(root=root, split="val", subset=True)
            test_set = MyZINC(root=root, split="test", subset=True)
        
        # =====================================================================
        # RANDOMIZED ZINC DATASETS FOR NULL MODEL EXPERIMENTS
        # =====================================================================
        elif name == "zinc_original":
            # Original ZINC - same as "zinc" but explicit for experiments
            from ..randomized_dataset import create_randomized_zinc_splits
            train_set, valid_set, test_set = create_randomized_zinc_splits(
                method="none",
                seed=seed,
                randomize_train_only=True,
                root=root,
            )
        
        elif name == "zinc_degree_preserving":
            # Degree-preserving randomization (train only)
            from ..randomized_dataset import create_randomized_zinc_splits
            train_set, valid_set, test_set = create_randomized_zinc_splits(
                method="degree_preserving",
                seed=seed,
                randomize_train_only=True,
                root=root,
            )
        
        elif name == "zinc_random":
            # Complete random rewiring (train only)
            from ..randomized_dataset import create_randomized_zinc_splits
            train_set, valid_set, test_set = create_randomized_zinc_splits(
                method="random",
                seed=seed,
                randomize_train_only=True,
                root=root,
            )
        
        elif name == "zinc_shuffled_node_features":
            # Shuffled node features (train only)
            from ..randomized_dataset import create_randomized_zinc_splits
            train_set, valid_set, test_set = create_randomized_zinc_splits(
                method="node_features",
                seed=seed,
                randomize_train_only=True,
                root=root,
            )
        
        elif name == "zinc_shuffled_edge_features":
            # Shuffled edge features (train only)
            from ..randomized_dataset import create_randomized_zinc_splits
            train_set, valid_set, test_set = create_randomized_zinc_splits(
                method="edge_features",
                seed=seed,
                randomize_train_only=True,
                root=root,
            )
        
        # =====================================================================
        # STOCHASTIC BLOCK MODEL (SBM) SYNTHETIC DATASETS
        # Now uses Degree-Corrected SBM (DCSBM) matching GraphWorld
        # =====================================================================
        
        elif name == "sbm" or name.startswith("sbm"):
            from ..dcsbm_dataset import create_dcsbm_splits
            
            # Parse parameters from dataset_spec (not name, which has params stripped)
            # Format: sbm or sbm:h=5.0,d=6.0,k=2,n=50-100,nf=constant,ef=constant,target=tailed_triangles
            # New DCSBM params: slope (cluster_size_slope), pexp (power_exponent), mindeg (min_deg)
            default_params = {
                "p_to_q_ratio": 5.0,       # h parameter (homophily)
                "avg_degree": 6.0,          # d parameter
                "num_communities": 2,       # k parameter
                "num_nodes_range": (50, 100),  # n parameter
                "num_train": 800,
                "num_val": 100,
                "num_test": 100,
                "node_feature_type": "community",
                "edge_feature_type": "community",
                "target_type": "tailed_triangles",
                # New DCSBM parameters
                "cluster_size_slope": 0.0,  # slope: 0=equal sizes, >0=imbalanced
                "power_exponent": 0.0,      # pexp: 0=uniform degrees, >0=power-law
                "min_deg": 1,               # mindeg: minimum degree for power-law
            }
            
            # Use dataset_spec (original input) to check for parameters
            if ":" in dataset_spec:
                param_str = dataset_spec.split(":")[1]
                for param in param_str.split(","):
                    if "=" not in param:
                        continue
                    key, value = param.split("=")
                    if key == "h":
                        default_params["p_to_q_ratio"] = float(value)
                    elif key == "d":
                        default_params["avg_degree"] = float(value)
                    elif key == "k":
                        default_params["num_communities"] = int(value)
                    elif key == "n":
                        if "-" in value:
                            low, high = value.split("-")
                            default_params["num_nodes_range"] = (int(low), int(high))
                        else:
                            n = int(value)
                            default_params["num_nodes_range"] = (n, n)
                    elif key == "train":
                        default_params["num_train"] = int(value)
                    elif key == "val":
                        default_params["num_val"] = int(value)
                    elif key == "test":
                        default_params["num_test"] = int(value)
                    elif key == "nf":
                        # Node feature type: community, constant, degree
                        default_params["node_feature_type"] = value
                    elif key == "ef":
                        # Edge feature type: community, constant
                        default_params["edge_feature_type"] = value
                    elif key == "target":
                        # Target type: tailed_triangles, triangle_count, edge_count
                        default_params["target_type"] = value
                    # New DCSBM parameters
                    elif key == "slope":
                        default_params["cluster_size_slope"] = float(value)
                    elif key == "pexp":
                        default_params["power_exponent"] = float(value)
                    elif key == "mindeg":
                        default_params["min_deg"] = int(value)
            
            print(f"[DCSBM] Creating dataset with params: {default_params}")
            
            train_set, valid_set, test_set = create_dcsbm_splits(
                num_train=default_params["num_train"],
                num_val=default_params["num_val"],
                num_test=default_params["num_test"],
                num_nodes_range=default_params["num_nodes_range"],
                num_communities=default_params["num_communities"],
                p_to_q_ratio=default_params["p_to_q_ratio"],
                avg_degree=default_params["avg_degree"],
                cluster_size_slope=default_params["cluster_size_slope"],
                power_exponent=default_params["power_exponent"],
                min_deg=default_params["min_deg"],
                node_feature_type=default_params["node_feature_type"],
                edge_feature_type=default_params["edge_feature_type"],
                target_type=default_params["target_type"],
                seed=seed,
            )
        
        # =====================================================================
        # END SYNTHETIC DATASETS
        # =====================================================================
        
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )