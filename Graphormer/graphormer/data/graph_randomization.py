# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Graph Randomization Module for Null Model Analysis

This module implements various graph randomization strategies to test
how much the model relies on graph structure vs. node/edge features.

Null Models Implemented:
1. Edge Shuffle (no preservation): Completely random rewiring
2. Degree-Preserving Randomization: Configuration model style rewiring
3. Node Feature Shuffle: Keep structure, randomize node features
4. Edge Feature Shuffle: Keep structure, randomize edge features

Reference: Maslov, S., & Sneppen, K. (2002). Specificity and stability 
in topology of protein networks. Science, 296(5569), 910-913.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple, Literal
import copy


def _edge_list_to_set(edge_index: torch.Tensor) -> set:
    """Convert edge_index to set of tuples for O(1) lookup."""
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def _has_edge(edge_set: set, u: int, v: int, directed: bool = False) -> bool:
    """Check if edge exists."""
    if directed:
        return (u, v) in edge_set
    return (u, v) in edge_set or (v, u) in edge_set


def shuffle_edges_random(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    seed: Optional[int] = None,
    keep_self_loops: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Completely random edge rewiring (Erdos-Renyi style null model).
    
    Each edge (u, v) is replaced with a random edge (u', v') where u' and v'
    are uniformly sampled from all nodes. Edge features stay with their
    original edge index position.
    
    This destroys:
    - Degree distribution
    - Clustering
    - All structural properties
    
    Args:
        edge_index: [2, E] tensor of edges
        edge_attr: [E] or [E, F] tensor of edge features (preserved per edge position)
        num_nodes: Number of nodes (inferred from edge_index if None)
        seed: Random seed for reproducibility
        keep_self_loops: If False, avoid creating self-loops
        
    Returns:
        new_edge_index: [2, E] randomized edges
        edge_attr: unchanged (same order as input)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    num_edges = edge_index.size(1)
    
    # Generate random edges
    new_src = np.random.randint(0, num_nodes, size=num_edges)
    new_dst = np.random.randint(0, num_nodes, size=num_edges)
    
    if not keep_self_loops:
        # Resample self-loops
        self_loop_mask = new_src == new_dst
        while self_loop_mask.any():
            new_dst[self_loop_mask] = np.random.randint(0, num_nodes, size=self_loop_mask.sum())
            self_loop_mask = new_src == new_dst
    
    new_edge_index = torch.tensor([new_src, new_dst], dtype=edge_index.dtype)
    
    return new_edge_index, edge_attr


def shuffle_edges_degree_preserving(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_swaps: Optional[int] = None,
    seed: Optional[int] = None,
    directed: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Degree-preserving edge randomization (Configuration model / Maslov-Sneppen).
    
    Performs edge swaps that preserve the degree of each node:
    - Pick two edges (a, b) and (c, d)
    - Swap to create (a, d) and (c, b)
    - Only accept if no multi-edges or self-loops are created
    
    This preserves:
    - Degree sequence (exact degree of each node)
    
    This destroys:
    - Clustering / triangles
    - Degree correlations (assortativity)
    - Community structure
    - Shortest path distribution (partially)
    
    Args:
        edge_index: [2, E] tensor of edges
        edge_attr: [E] or [E, F] tensor of edge features (swapped WITH edges)
        num_swaps: Number of swap attempts (default: 10 * num_edges)
        seed: Random seed
        directed: Whether graph is directed
        
    Returns:
        new_edge_index: [2, E] randomized edges (same degrees)
        new_edge_attr: edge features (swapped with their edges)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Work with numpy for efficiency
    edges = edge_index.numpy().T.copy()  # [E, 2]
    num_edges = len(edges)
    
    # Handle edge attributes
    if edge_attr is not None:
        attrs = edge_attr.clone()
    else:
        attrs = None
    
    if num_swaps is None:
        num_swaps = 10 * num_edges
    
    # For undirected graphs, we need to handle edge pairs
    if not directed:
        # Find unique undirected edges (assuming edges appear as both (u,v) and (v,u))
        edge_set = set()
        unique_edge_indices = []
        for i, (u, v) in enumerate(edges):
            if (min(u, v), max(u, v)) not in edge_set:
                edge_set.add((min(u, v), max(u, v)))
                unique_edge_indices.append(i)
        
        # Work with unique edges only
        work_indices = np.array(unique_edge_indices)
    else:
        work_indices = np.arange(num_edges)
    
    # Build edge set for O(1) existence checks
    current_edge_set = _edge_list_to_set(torch.tensor(edges.T))
    
    for _ in range(num_swaps):
        # Pick two random edge indices
        if len(work_indices) < 2:
            break
            
        idx1, idx2 = np.random.choice(len(work_indices), size=2, replace=False)
        e1_idx, e2_idx = work_indices[idx1], work_indices[idx2]
        
        a, b = edges[e1_idx]
        c, d = edges[e2_idx]
        
        # Skip if edges share a node
        if len({a, b, c, d}) < 4:
            continue
        
        # Proposed new edges: (a, d) and (c, b)
        # Check for self-loops
        if a == d or c == b:
            continue
        
        # Check if new edges already exist (would create multi-edge)
        if _has_edge(current_edge_set, a, d, directed) or _has_edge(current_edge_set, c, b, directed):
            continue
        
        # Perform the swap
        # Remove old edges from set
        current_edge_set.discard((a, b))
        current_edge_set.discard((c, d))
        if not directed:
            current_edge_set.discard((b, a))
            current_edge_set.discard((d, c))
        
        # Add new edges to set
        current_edge_set.add((a, d))
        current_edge_set.add((c, b))
        if not directed:
            current_edge_set.add((d, a))
            current_edge_set.add((b, c))
        
        # Update edge array
        edges[e1_idx] = [a, d]
        edges[e2_idx] = [c, b]
        
        # For undirected: also update reverse edges
        if not directed:
            for i, (u, v) in enumerate(edges):
                if u == b and v == a:
                    edges[i] = [d, a]
                    if attrs is not None:
                        attrs[i] = attrs[e1_idx].clone()
                elif u == d and v == c:
                    edges[i] = [b, c]
                    if attrs is not None:
                        attrs[i] = attrs[e2_idx].clone()
        
        # Swap edge attributes
        if attrs is not None:
            attrs[e1_idx], attrs[e2_idx] = attrs[e2_idx].clone(), attrs[e1_idx].clone()
    
    new_edge_index = torch.tensor(edges.T, dtype=edge_index.dtype)
    
    return new_edge_index, attrs


def shuffle_node_features(
    x: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Shuffle node features while keeping graph structure intact.
    
    Args:
        x: [N] or [N, F] node features
        seed: Random seed
        
    Returns:
        Shuffled node features (same shape as input)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes)
    return x[perm]


def shuffle_edge_features(
    edge_attr: torch.Tensor,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Shuffle edge features while keeping graph structure intact.
    
    This keeps edges in place but randomly reassigns their features.
    
    Args:
        edge_attr: [E] or [E, F] edge features
        seed: Random seed
        
    Returns:
        Shuffled edge features (same shape as input)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_edges = edge_attr.size(0)
    perm = torch.randperm(num_edges)
    return edge_attr[perm]


def randomize_graph(
    data: Data,
    method: Literal[
        "none",
        "random",
        "degree_preserving", 
        "node_features",
        "edge_features",
    ] = "degree_preserving",
    seed: Optional[int] = None,
    **kwargs,
) -> Data:
    """
    Apply randomization to a PyG Data object.
    
    Args:
        data: PyG Data object with edge_index, edge_attr, x, y
        method: Randomization method:
            - "none": No randomization (return copy of original)
            - "random": Complete random rewiring (destroys all structure)
            - "degree_preserving": Preserve degree sequence
            - "node_features": Keep structure, shuffle node features
            - "edge_features": Keep structure, shuffle edge features  
        seed: Random seed
        **kwargs: Additional arguments for specific methods
        
    Returns:
        New Data object with randomization applied
    """
    # Deep copy to avoid modifying original
    new_data = Data()
    for key in data.keys():
        if torch.is_tensor(data[key]):
            new_data[key] = data[key].clone()
        else:
            new_data[key] = copy.deepcopy(data[key])
    
    # If method is "none", just return the copy
    if method == "none":
        return new_data
    
    num_nodes = new_data.x.size(0) if new_data.x is not None else new_data.edge_index.max().item() + 1
    
    if method == "random":
        new_edge_index, new_edge_attr = shuffle_edges_random(
            new_data.edge_index,
            new_data.edge_attr,
            num_nodes=num_nodes,
            seed=seed,
            **kwargs,
        )
        new_data.edge_index = new_edge_index
        if new_edge_attr is not None:
            new_data.edge_attr = new_edge_attr
            
    elif method == "degree_preserving":
        new_edge_index, new_edge_attr = shuffle_edges_degree_preserving(
            new_data.edge_index,
            new_data.edge_attr,
            seed=seed,
            **kwargs,
        )
        new_data.edge_index = new_edge_index
        if new_edge_attr is not None:
            new_data.edge_attr = new_edge_attr
        
    elif method == "node_features":
        new_data.x = shuffle_node_features(new_data.x, seed=seed)
        
    elif method == "edge_features":
        if new_data.edge_attr is not None:
            new_data.edge_attr = shuffle_edge_features(new_data.edge_attr, seed=seed)
        
    else:
        raise ValueError(f"Unknown randomization method: {method}")
    
    return new_data


class RandomizedDatasetWrapper:
    """
    Wrapper that applies randomization to a dataset on-the-fly.
    
    This allows testing null models without creating new dataset files.
    
    Example:
        >>> dataset = MyPYGDataset(root='./data')
        >>> null_dataset = RandomizedDatasetWrapper(
        ...     dataset, 
        ...     method="degree_preserving",
        ...     seed=42
        ... )
        >>> null_dataset[0]  # Returns randomized graph
    """
    
    def __init__(
        self,
        dataset,
        method: str = "degree_preserving",
        seed: Optional[int] = None,
        randomize_once: bool = True,
        **kwargs,
    ):
        """
        Args:
            dataset: Base PyG dataset
            method: Randomization method (see randomize_graph)
            seed: Base random seed (graph i uses seed + i)
            randomize_once: If True, cache randomized graphs
            **kwargs: Arguments passed to randomize_graph
        """
        self.dataset = dataset
        self.method = method
        self.base_seed = seed
        self.randomize_once = randomize_once
        self.kwargs = kwargs
        self._cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.randomize_once and idx in self._cache:
            return self._cache[idx]
        
        data = self.dataset[idx]
        seed = self.base_seed + idx if self.base_seed is not None else None
        
        randomized = randomize_graph(data, method=self.method, seed=seed, **self.kwargs)
        
        if self.randomize_once:
            self._cache[idx] = randomized
        
        return randomized
    
    def clear_cache(self):
        self._cache = {}