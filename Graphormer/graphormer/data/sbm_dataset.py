# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Stochastic Block Model (SBM) Synthetic Dataset for Graphormer

This module generates synthetic graphs using the Stochastic Block Model
with controllable homophily (p/q ratio) and average degree.

The SBM is parameterized by:
- n: number of nodes per graph
- k: number of communities/blocks  
- p: probability of edge within same community (intra-community)
- q: probability of edge between different communities (inter-community)

Homophily ratio = p/q
- High p/q (e.g., 10): Strong community structure (homophilic)
- p/q = 1: Random graph (no community structure)
- Low p/q (e.g., 0.1): Heterophilic structure

Tasks supported:
- Community count prediction (regression)
- Graph classification by homophily level
- Node feature aggregation prediction
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset, InMemoryDataset
from typing import Optional, Tuple, List, Literal
import os
import os.path as osp
from tqdm import tqdm


def count_triangles(edge_index: torch.Tensor, num_nodes: int) -> int:
    """
    Count the number of triangles in an undirected graph.
    
    Uses adjacency matrix multiplication: trace(A^3) / 6
    """
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    
    # A^3 diagonal elements count paths of length 3 from node to itself
    # Each triangle is counted 6 times (3 nodes × 2 directions)
    adj2 = torch.mm(adj, adj)
    adj3 = torch.mm(adj2, adj)
    num_triangles = int(torch.trace(adj3).item() / 6)
    
    return num_triangles


def count_tailed_triangles(edge_index: torch.Tensor, num_nodes: int) -> int:
    """
    Count the number of tailed triangles in an undirected graph (fully vectorized).
    
    A tailed triangle is a triangle with a 4th "tail" node connected to exactly
    one vertex of the triangle. We count all (triangle, tail_node) pairs.
    
    Returns:
        Number of tailed triangles
    """
    if num_nodes < 3:
        return 0
        
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    
    # Degrees of each node
    degrees = adj.sum(dim=1)
    
    # Count triangles per node using A³ diagonal
    adj2 = torch.mm(adj, adj)
    adj3_diag = (adj2 * adj).sum(dim=1)  # Faster than full A³
    
    # triangles_per_node[i] = number of triangles containing node i (each counted once)
    triangles_per_node = adj3_diag / 2
    
    total_triangles = int(triangles_per_node.sum().item() / 3)
    
    if total_triangles == 0:
        return 0
    
    # For each triangle containing node i, node i contributes (degree_i - 2) tails
    # because 2 of its edges are used by the triangle, the rest are tails
    # 
    # Total tailed triangles = sum over all nodes i:
    #   triangles_per_node[i] * (degree[i] - 2)
    
    tails_per_node = triangles_per_node * (degrees - 2)
    tailed_count = int(tails_per_node.sum().item())
    
    # Clamp to non-negative (in case of numerical issues)
    return max(0, tailed_count)


def compute_triangle_density(edge_index: torch.Tensor, num_nodes: int) -> float:
    """
    Compute triangle density: number of triangles / number of possible triangles.
    
    Possible triangles = C(n, 3) = n * (n-1) * (n-2) / 6
    """
    num_triangles = count_triangles(edge_index, num_nodes)
    
    # Number of possible triangles (combinations of 3 nodes)
    if num_nodes < 3:
        return 0.0
    
    possible_triangles = num_nodes * (num_nodes - 1) * (num_nodes - 2) / 6
    
    if possible_triangles == 0:
        return 0.0
    
    return num_triangles / possible_triangles


def generate_sbm_graph(
    num_nodes: int = 50,
    num_communities: int = 2,
    p: float = 0.3,
    q: float = 0.05,
    node_feature_dim: int = 1,
    node_feature_type: Literal["community", "random", "degree", "one_hot", "constant"] = "community",
    edge_feature_type: Literal["community", "random", "constant"] = "community",
    target_type: Literal["tailed_triangles", "triangle_count", "triangle_density", "edge_count"] = "tailed_triangles",
    seed: Optional[int] = None,
) -> Data:
    """
    Generate a single SBM graph.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_communities: Number of communities/blocks (k)
        p: Intra-community edge probability
        q: Inter-community edge probability
        node_feature_dim: Dimension of node features (used for "random" type)
        node_feature_type: How to generate node features:
            - "community": Community label as feature
            - "random": Random features
            - "degree": Degree as feature (computed after graph generation)
            - "one_hot": One-hot encoding of community
            - "constant": All nodes have the same feature (value=1)
        edge_feature_type: How to generate edge features:
            - "community": 1 for intra-community, 2 for inter-community
            - "random": Random edge features
            - "constant": All edges have the same feature (value=1)
        target_type: What to predict:
            - "tailed_triangles": Count of tailed triangles
            - "triangle_count": Number of triangles
            - "triangle_density": Triangles / possible triangles
            - "edge_count": Number of edges
        seed: Random seed
        
    Returns:
        PyG Data object with the generated graph
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Assign nodes to communities (roughly equal sizes)
    community_sizes = [num_nodes // num_communities] * num_communities
    # Distribute remainder
    for i in range(num_nodes % num_communities):
        community_sizes[i] += 1
    
    community_labels = []
    for c, size in enumerate(community_sizes):
        community_labels.extend([c] * size)
    community_labels = np.array(community_labels)
    
    # Generate edges based on SBM
    edges_src = []
    edges_dst = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Determine edge probability
            if community_labels[i] == community_labels[j]:
                prob = p  # Same community
            else:
                prob = q  # Different communities
            
            # Sample edge
            if np.random.random() < prob:
                # Add both directions (undirected graph)
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
    
    # Handle case of no edges
    if len(edges_src) == 0:
        # Add at least one edge to avoid issues
        edges_src = [0, 1]
        edges_dst = [1, 0]
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # Generate node features
    if node_feature_type == "community":
        # Community label as feature
        x = torch.tensor(community_labels, dtype=torch.long).unsqueeze(-1)
    elif node_feature_type == "one_hot":
        # One-hot encoding of community
        x = torch.zeros(num_nodes, num_communities, dtype=torch.long)
        for i, c in enumerate(community_labels):
            x[i, c] = 1
    elif node_feature_type == "random":
        # Random features
        x = torch.randint(0, 10, (num_nodes, node_feature_dim), dtype=torch.long)
    elif node_feature_type == "degree":
        # Degree as feature
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        x = degrees.unsqueeze(-1).long()
    elif node_feature_type == "constant":
        # All nodes have the same feature
        x = torch.ones(num_nodes, 1, dtype=torch.long)
    else:
        raise ValueError(f"Unknown node_feature_type: {node_feature_type}")
    
    # Generate edge features
    num_edges = edge_index.size(1)
    if edge_feature_type == "community":
        # Intra vs inter community indicator
        edge_attr = torch.zeros(num_edges, dtype=torch.long)
        for e in range(num_edges):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            if community_labels[src] == community_labels[dst]:
                edge_attr[e] = 1  # Intra-community edge
            else:
                edge_attr[e] = 2  # Inter-community edge
    elif edge_feature_type == "random":
        # Random edge features
        edge_attr = torch.randint(1, 5, (num_edges,), dtype=torch.long)
    elif edge_feature_type == "constant":
        # All edges have the same feature
        edge_attr = torch.ones(num_edges, dtype=torch.long)
    else:
        raise ValueError(f"Unknown edge_feature_type: {edge_feature_type}")
    
    # Compute graph statistics
    num_triangles = count_triangles(edge_index, num_nodes)
    num_tailed_triangles = count_tailed_triangles(edge_index, num_nodes)
    triangle_density = compute_triangle_density(edge_index, num_nodes)
    num_edges = edge_index.size(1) // 2  # Undirected
    
    # Set target based on target_type
    if target_type == "tailed_triangles":
        y_value = float(num_tailed_triangles)
    elif target_type == "triangle_count":
        y_value = float(num_triangles)
    elif target_type == "triangle_density":
        y_value = triangle_density
    elif target_type == "edge_count":
        y_value = float(num_edges)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    y = torch.tensor([[y_value]], dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
    )
    
    # Store metadata
    data.community_labels = torch.tensor(community_labels, dtype=torch.long)
    data.num_communities = num_communities
    data.p = p
    data.q = q
    data.num_triangles = num_triangles
    data.num_tailed_triangles = num_tailed_triangles
    data.triangle_density = triangle_density
    data.target_type = target_type
    
    return data


def compute_p_q_from_homophily_and_degree(
    num_nodes: int,
    num_communities: int,
    homophily_ratio: float,
    avg_degree: float,
) -> Tuple[float, float]:
    """
    Compute p and q from desired homophily ratio and average degree.
    
    Homophily ratio h = p/q
    Average degree d ≈ (n/k - 1) * p + (n - n/k) * q
    
    Where n = num_nodes, k = num_communities
    
    Args:
        num_nodes: Number of nodes
        num_communities: Number of communities
        homophily_ratio: Desired p/q ratio (>1 for homophilic, <1 for heterophilic)
        avg_degree: Desired average degree
        
    Returns:
        (p, q) tuple
    """
    n = num_nodes
    k = num_communities
    h = homophily_ratio
    d = avg_degree
    
    # Nodes per community (approximate)
    n_c = n / k
    
    # Number of possible intra-community edges per node: n_c - 1
    # Number of possible inter-community edges per node: n - n_c
    intra_possible = n_c - 1
    inter_possible = n - n_c
    
    # Average degree: d = intra_possible * p + inter_possible * q
    # Homophily: h = p / q, so p = h * q
    # Substituting: d = intra_possible * h * q + inter_possible * q
    # d = q * (intra_possible * h + inter_possible)
    # q = d / (intra_possible * h + inter_possible)
    
    q = d / (intra_possible * h + inter_possible)
    p = h * q
    
    # Clamp to valid probability range
    p = min(max(p, 0.001), 0.999)
    q = min(max(q, 0.001), 0.999)
    
    return p, q


class SBMDataset(InMemoryDataset):
    """
    Stochastic Block Model synthetic dataset for Graphormer.
    
    Generates graphs with controllable community structure for
    testing GNN sensitivity to graph topology.
    
    Example:
        >>> dataset = SBMDataset(
        ...     root='./dataset/sbm',
        ...     num_graphs=1000,
        ...     num_nodes=50,
        ...     num_communities=2,
        ...     homophily_ratio=5.0,
        ...     avg_degree=6.0,
        ... )
    """
    
    def __init__(
        self,
        root: str,
        num_graphs: int = 10000,
        num_nodes: int = 50,
        num_communities: int = 2,
        homophily_ratio: float = 5.0,
        avg_degree: float = 6.0,
        node_feature_type: str = "community",
        target_type: Literal["homophily", "community_count", "avg_degree"] = "homophily",
        split: Literal["train", "val", "test"] = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.homophily_ratio = homophily_ratio
        self.avg_degree = avg_degree
        self.node_feature_type = node_feature_type
        self.target_type = target_type
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        # Compute p and q
        self.p, self.q = compute_p_q_from_homophily_and_degree(
            num_nodes, num_communities, homophily_ratio, avg_degree
        )
        
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        # Include parameters in filename to avoid conflicts
        name = (f"sbm_n{self.num_nodes}_k{self.num_communities}_"
                f"h{self.homophily_ratio}_d{self.avg_degree}_"
                f"{self.split}_s{self.seed}.pt")
        return [name]
    
    def download(self):
        pass  # No download needed
    
    def process(self):
        np.random.seed(self.seed)
        
        # Determine split sizes
        n_train = int(self.num_graphs * self.train_ratio)
        n_val = int(self.num_graphs * self.val_ratio)
        n_test = self.num_graphs - n_train - n_val
        
        if self.split == "train":
            n_graphs = n_train
            offset = 0
        elif self.split == "val":
            n_graphs = n_val
            offset = n_train
        else:  # test
            n_graphs = n_test
            offset = n_train + n_val
        
        print(f"[SBMDataset] Generating {n_graphs} graphs for {self.split} split...")
        print(f"[SBMDataset] Parameters: n={self.num_nodes}, k={self.num_communities}, "
              f"p={self.p:.4f}, q={self.q:.4f}, h={self.homophily_ratio}, d={self.avg_degree}")
        
        data_list = []
        for i in tqdm(range(n_graphs), desc=f"Generating {self.split}"):
            graph_seed = self.seed + offset + i
            
            # Add some variation in graph size
            num_nodes = self.num_nodes + np.random.randint(-10, 11)
            num_nodes = max(20, min(100, num_nodes))  # Clamp to 20-100
            
            # Recompute p, q for this graph size
            p, q = compute_p_q_from_homophily_and_degree(
                num_nodes, self.num_communities, self.homophily_ratio, self.avg_degree
            )
            
            data = generate_sbm_graph(
                num_nodes=num_nodes,
                num_communities=self.num_communities,
                p=p,
                q=q,
                node_feature_type=self.node_feature_type,
                seed=graph_seed,
            )
            
            # Set target based on target_type
            if self.target_type == "homophily":
                # Already set in generate_sbm_graph
                pass
            elif self.target_type == "community_count":
                data.y = torch.tensor([[self.num_communities]], dtype=torch.float)
            elif self.target_type == "avg_degree":
                actual_avg_degree = data.edge_index.size(1) / data.num_nodes
                data.y = torch.tensor([[actual_avg_degree]], dtype=torch.float)
            
            data_list.append(data)
        
        self.save(data_list, self.processed_paths[0])


class SBMDatasetSimple:
    """
    Simple SBM dataset that doesn't use InMemoryDataset (generates on-the-fly).
    Useful for quick experiments without disk caching.
    """
    
    def __init__(
        self,
        num_graphs: int = 1000,
        num_nodes_range: Tuple[int, int] = (50, 100),
        num_communities: int = 2,
        homophily_ratio: float = 5.0,
        avg_degree: float = 6.0,
        node_feature_type: str = "community",
        edge_feature_type: str = "community",
        target_type: str = "tailed_triangles",
        seed: int = 42,
    ):
        self.num_graphs = num_graphs
        self.num_nodes_range = num_nodes_range
        self.num_communities = num_communities
        self.homophily_ratio = homophily_ratio
        self.avg_degree = avg_degree
        self.node_feature_type = node_feature_type
        self.edge_feature_type = edge_feature_type
        self.target_type = target_type
        self.seed = seed
        self._cache = {}
    
    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        
        np.random.seed(self.seed + idx)
        
        # Random number of nodes in range
        num_nodes = np.random.randint(self.num_nodes_range[0], self.num_nodes_range[1] + 1)
        
        p, q = compute_p_q_from_homophily_and_degree(
            num_nodes, self.num_communities, self.homophily_ratio, self.avg_degree
        )
        
        data = generate_sbm_graph(
            num_nodes=num_nodes,
            num_communities=self.num_communities,
            p=p,
            q=q,
            node_feature_type=self.node_feature_type,
            edge_feature_type=self.edge_feature_type,
            target_type=self.target_type,
            seed=self.seed + idx,
        )
        
        self._cache[idx] = data
        return data


def create_sbm_splits(
    num_train: int = 800,
    num_val: int = 100,
    num_test: int = 100,
    num_nodes_range: Tuple[int, int] = (50, 100),
    num_communities: int = 2,
    homophily_ratio: float = 5.0,
    avg_degree: float = 6.0,
    node_feature_type: str = "community",
    edge_feature_type: str = "community",
    target_type: str = "tailed_triangles",
    seed: int = 42,
) -> Tuple[SBMDatasetSimple, SBMDatasetSimple, SBMDatasetSimple]:
    """
    Create train/val/test splits for SBM dataset.
    
    Default: 800 train, 100 val, 100 test (1000 total)
    
    Args:
        num_train: Number of training graphs
        num_val: Number of validation graphs
        num_test: Number of test graphs
        num_nodes_range: (min, max) node count per graph
        num_communities: Number of communities (k)
        homophily_ratio: p/q ratio
        avg_degree: Target average degree
        node_feature_type: "community", "random", "degree", "one_hot", or "constant"
        edge_feature_type: "community", "random", or "constant"
        target_type: "tailed_triangles", "triangle_count", "triangle_density", or "edge_count"
        seed: Random seed
    
    Returns:
        Tuple of (train_set, valid_set, test_set)
    """
    train_set = SBMDatasetSimple(
        num_graphs=num_train,
        num_nodes_range=num_nodes_range,
        num_communities=num_communities,
        homophily_ratio=homophily_ratio,
        avg_degree=avg_degree,
        node_feature_type=node_feature_type,
        edge_feature_type=edge_feature_type,
        target_type=target_type,
        seed=seed,
    )
    
    valid_set = SBMDatasetSimple(
        num_graphs=num_val,
        num_nodes_range=num_nodes_range,
        num_communities=num_communities,
        homophily_ratio=homophily_ratio,
        avg_degree=avg_degree,
        node_feature_type=node_feature_type,
        edge_feature_type=edge_feature_type,
        target_type=target_type,
        seed=seed + num_train,
    )
    
    test_set = SBMDatasetSimple(
        num_graphs=num_test,
        num_nodes_range=num_nodes_range,
        num_communities=num_communities,
        homophily_ratio=homophily_ratio,
        avg_degree=avg_degree,
        node_feature_type=node_feature_type,
        edge_feature_type=edge_feature_type,
        target_type=target_type,
        seed=seed + num_train + num_val,
    )
    
    print(f"[SBMDataset] Created splits: {num_train} train, {num_val} val, {num_test} test")
    print(f"[SBMDataset] Parameters: nodes={num_nodes_range}, k={num_communities}, "
          f"homophily={homophily_ratio}, avg_degree={avg_degree}")
    print(f"[SBMDataset] Node features: {node_feature_type}, Edge features: {edge_feature_type}")
    print(f"[SBMDataset] Target: {target_type}")
    
    return train_set, valid_set, test_set


# Quick test
if __name__ == "__main__":
    print("Testing SBM graph generation...")
    
    # Test single graph
    data = generate_sbm_graph(
        num_nodes=50,
        num_communities=2,
        p=0.3,
        q=0.05,
        seed=42,
    )
    
    print(f"Generated graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Avg degree: {data.edge_index.size(1) / data.num_nodes:.2f}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Target (homophily): {data.y.item():.3f}")
    print(f"  Community distribution: {torch.bincount(data.community_labels).tolist()}")
    
    # Test p,q computation
    print("\nTesting p,q computation...")
    for h in [0.5, 1.0, 2.0, 5.0, 10.0]:
        p, q = compute_p_q_from_homophily_and_degree(50, 2, h, 6.0)
        print(f"  homophily={h}: p={p:.4f}, q={q:.4f}, ratio={p/q:.2f}")
    
    # Test dataset creation
    print("\nTesting dataset splits...")
    train, val, test = create_sbm_splits(
        num_train=100,
        num_val=20,
        num_test=20,
        homophily_ratio=5.0,
        avg_degree=6.0,
    )
    
    print(f"Train size: {len(train)}")
    print(f"Sample train graph: nodes={train[0].num_nodes}, edges={train[0].edge_index.size(1)}")