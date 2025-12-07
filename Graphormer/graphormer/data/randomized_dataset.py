# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Randomized Dataset Wrapper for Graphormer Null Model Analysis

This module provides a simple wrapper that randomizes graphs on-the-fly
while being compatible with Graphormer's data pipeline.
"""

from torch_geometric.data import Dataset
from typing import Optional, Literal, Tuple
import torch
import copy

from .graph_randomization import randomize_graph


class RandomizedPYGDataset(Dataset):
    """
    A simple wrapper that applies randomization to a PyG dataset.
    
    This wrapper is designed to work with GraphormerPYGDataset's create_subset method.
    
    Example:
        >>> from torch_geometric.datasets import ZINC
        >>> base = ZINC(root='./data', split='train')
        >>> randomized = RandomizedPYGDataset(base, method="degree_preserving")
        >>> randomized[0]  # Returns randomized graph
    """
    
    def __init__(
        self,
        dataset: Dataset,
        randomization_method: str = "none",
        seed: int = 0,
        cache: bool = False,
        **randomization_kwargs,
    ):
        """
        Args:
            dataset: Base PyG dataset
            randomization_method: How to randomize graphs
            seed: Random seed (graph i uses seed + i)
            cache: Whether to cache randomized graphs
            **randomization_kwargs: Additional args for randomization
        """
        super().__init__()
        self.dataset = dataset
        self.randomization_method = randomization_method
        self.base_seed = seed
        self.cache = cache
        self.randomization_kwargs = randomization_kwargs
        self._cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Check cache first
            if self.cache and idx in self._cache:
                return self._cache[idx]
            
            # Get original item
            item = self.dataset[idx]
            
            # Apply randomization if needed
            if self.randomization_method != "none":
                seed = self.base_seed + idx
                item = randomize_graph(
                    item,
                    method=self.randomization_method,
                    seed=seed,
                    **self.randomization_kwargs,
                )
            
            # Cache if enabled
            if self.cache:
                self._cache[idx] = item
            
            return item
        else:
            raise TypeError("Index must be an integer")
    
    def index_select(self, idx):
        """
        Create a subset of this dataset.
        Required for compatibility with GraphormerPYGDataset.
        """
        return RandomizedPYGDatasetSubset(self, idx)
    
    def get(self, idx):
        """Alternative access method used by some PyG code."""
        return self.__getitem__(idx)


class RandomizedPYGDatasetSubset(Dataset):
    """
    A subset of a RandomizedPYGDataset.
    Created by index_select() for compatibility with GraphormerPYGDataset.
    """
    
    def __init__(self, parent: RandomizedPYGDataset, indices):
        super().__init__()
        self.parent = parent
        if isinstance(indices, torch.Tensor):
            self.indices = indices.tolist()
        else:
            self.indices = list(indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            actual_idx = self.indices[idx]
            return self.parent[actual_idx]
        else:
            raise TypeError("Index must be an integer")
    
    def index_select(self, idx):
        """Nested index_select for further subsetting."""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        new_indices = [self.indices[i] for i in idx]
        return RandomizedPYGDatasetSubset(self.parent, new_indices)
    
    def get(self, idx):
        return self.__getitem__(idx)


def create_randomized_zinc_splits(
    method: str = "none",
    seed: int = 0,
    randomize_train_only: bool = True,
    root: str = "dataset",
    subset: bool = True,
) -> Tuple[RandomizedPYGDataset, RandomizedPYGDataset, RandomizedPYGDataset]:
    """
    Create randomized ZINC train/valid/test splits.
    
    This function is called by PYGDatasetLookupTable to create randomized
    versions of ZINC.
    
    Args:
        method: Randomization method:
            - "none": No randomization (original graphs)
            - "degree_preserving": Preserve degree sequence, randomize edges
            - "random": Completely random rewiring
            - "node_features": Shuffle node features, keep structure
            - "edge_features": Shuffle edge features, keep structure
        seed: Random seed for reproducibility
        randomize_train_only: If True, only randomize training set
        root: Root directory for dataset
        subset: Whether to use ZINC subset (True) or full (False)
    
    Returns:
        Tuple of (train_set, valid_set, test_set) with randomization applied
    """
    from torch_geometric.datasets import ZINC
    import torch.distributed as dist
    
    # Handle distributed downloading
    class DistributedZINC(ZINC):
        def download(self):
            if not dist.is_initialized() or dist.get_rank() == 0:
                super().download()
            if dist.is_initialized():
                dist.barrier()

        def process(self):
            if not dist.is_initialized() or dist.get_rank() == 0:
                super().process()
            if dist.is_initialized():
                dist.barrier()
    
    # Load base datasets
    print(f"[RandomizedDataset] Loading ZINC with method={method}, randomize_train_only={randomize_train_only}")
    
    base_train = DistributedZINC(root=root, split="train", subset=subset)
    base_valid = DistributedZINC(root=root, split="val", subset=subset)
    base_test = DistributedZINC(root=root, split="test", subset=subset)
    
    print(f"[RandomizedDataset] Loaded: {len(base_train)} train, {len(base_valid)} valid, {len(base_test)} test")
    
    # Wrap with randomization
    train_set = RandomizedPYGDataset(
        base_train,
        randomization_method=method,
        seed=seed,
    )
    
    if randomize_train_only:
        # Valid and test are NOT randomized
        valid_set = RandomizedPYGDataset(
            base_valid,
            randomization_method="none",
            seed=seed,
        )
        test_set = RandomizedPYGDataset(
            base_test,
            randomization_method="none",
            seed=seed,
        )
    else:
        # All splits are randomized
        valid_set = RandomizedPYGDataset(
            base_valid,
            randomization_method=method,
            seed=seed + 1000000,
        )
        test_set = RandomizedPYGDataset(
            base_test,
            randomization_method=method,
            seed=seed + 2000000,
        )
    
    if method != "none":
        if randomize_train_only:
            print(f"[RandomizedDataset] Train set will be randomized, valid/test unchanged")
        else:
            print(f"[RandomizedDataset] All splits will be randomized")
    
    return train_set, valid_set, test_set


# Backwards compatibility alias
GraphormerRandomizedPYGDataset = RandomizedPYGDataset