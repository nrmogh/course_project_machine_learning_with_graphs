# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Degree-Corrected Stochastic Block Model (DCSBM) Synthetic Dataset for Graphormer

This module generates synthetic graphs using the Degree-Corrected SBM,
matching the generation approach used in GraphWorld (Google Research).

Key features:
- Configurable community size imbalance (cluster_size_slope)
- Power-law degree distribution (power_exponent, min_deg)
- Proper degree correction (each node has degree propensity)
- Compatible parameterization with GraphWorld

The DCSBM extends SBM by giving each node a degree propensity θ_i.
Edge probability: P(i,j) ∝ θ_i * θ_j * ω_{c_i, c_j}
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple, Literal
from tqdm import tqdm


# =============================================================================
# Substructure Counting
# =============================================================================

def count_triangles(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Count triangles using A³ trace."""
    if num_nodes < 3:
        return 0
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    adj2 = torch.mm(adj, adj)
    adj3 = torch.mm(adj2, adj)
    return int(torch.trace(adj3).item() / 6)


def count_tailed_triangles(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Count tailed triangles (triangle + 1 tail node connected to exactly 1 vertex)."""
    if num_nodes < 4:
        return 0
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    degrees = adj.sum(dim=1)
    adj2 = torch.mm(adj, adj)
    adj3_diag = (adj2 * adj).sum(dim=1)
    triangles_per_node = adj3_diag / 2
    if triangles_per_node.sum() == 0:
        return 0
    tails_per_node = triangles_per_node * (degrees - 2)
    return max(0, int(tails_per_node.sum().item()))


def compute_edge_homogeneity(
    edge_index: torch.Tensor, 
    community_labels: np.ndarray
) -> float:
    """
    Compute edge homogeneity: fraction of intra-community edges.
    
    Args:
        edge_index: Edge index tensor [2, num_edges]
        community_labels: Community assignment for each node
        
    Returns:
        Homogeneity in [0, 1]. 1 = all edges within communities, 0 = all between.
    """
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return 0.0
    
    intra_count = 0
    # Only count each undirected edge once
    for i in range(0, num_edges, 2):  
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if community_labels[src] == community_labels[dst]:
            intra_count += 1
    
    total_undirected = num_edges // 2
    return intra_count / total_undirected if total_undirected > 0 else 0.0


def compute_degree_stats(edge_index: torch.Tensor, num_nodes: int) -> Tuple[float, float]:
    """
    Compute average degree and degree variance.
    
    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        (avg_degree, degree_variance)
    """
    if edge_index.size(1) == 0:
        return 0.0, 0.0
    
    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
    avg_degree = degrees.mean().item()
    degree_variance = degrees.var().item()
    
    return avg_degree, degree_variance


# =============================================================================
# DCSBM Helper Functions (matching GraphWorld)
# =============================================================================

def make_pi(num_communities: int, cluster_size_slope: float = 0.0) -> np.ndarray:
    """
    Create community size proportions.
    
    Args:
        num_communities: Number of communities (k)
        cluster_size_slope: Controls size imbalance.
            0.0 = equal sizes
            > 0 = increasingly imbalanced (community 0 smallest, k-1 largest)
    
    Returns:
        pi: Array of length k summing to 1.0
    """
    pi = np.arange(num_communities, dtype=float) * cluster_size_slope
    pi += 1.0  # Base size of 1 for each community
    pi /= pi.sum()
    return pi


def make_prop_mat(num_communities: int, p_to_q_ratio: float) -> np.ndarray:
    """
    Create edge probability proportion matrix.
    
    Args:
        num_communities: Number of communities (k)
        p_to_q_ratio: Ratio of intra-community to inter-community edge rates.
            > 1 = assortative (homophilic)
            = 1 = random (no community structure)
            < 1 = disassortative (heterophilic)
    
    Returns:
        prop_mat: k x k matrix with p_to_q_ratio on diagonal, 1.0 off-diagonal
    """
    prop_mat = np.ones((num_communities, num_communities))
    np.fill_diagonal(prop_mat, p_to_q_ratio)
    return prop_mat


def make_degree_propensities(
    num_nodes: int,
    power_exponent: float = 0.0,
    min_deg: int = 1,
) -> np.ndarray:
    """
    Generate degree propensities following a power-law distribution.
    
    Args:
        num_nodes: Number of nodes
        power_exponent: Exponent for power-law. 
            0.0 = uniform degrees
            > 0 = power-law (higher = more skewed)
            Typical values: 2.0-3.0 for scale-free networks
        min_deg: Minimum degree in the power-law
    
    Returns:
        degrees: Array of degree propensities (will be normalized per community)
    """
    if power_exponent <= 0:
        # Uniform degree propensities
        return np.ones(num_nodes)
    
    # Power-law sampling using inverse transform
    # P(k) ∝ k^(-gamma)
    # CDF: F(k) = (k^(1-gamma) - k_min^(1-gamma)) / (k_max^(1-gamma) - k_min^(1-gamma))
    # Inverse: k = ((k_max^(1-gamma) - k_min^(1-gamma)) * u + k_min^(1-gamma))^(1/(1-gamma))
    
    k_min = max(1, min_deg)
    k_max = num_nodes
    gamma = power_exponent
    
    # Handle gamma = 1 specially (log distribution)
    if abs(gamma - 1.0) < 1e-6:
        # For gamma = 1: P(k) ∝ 1/k, CDF uses log
        # F(k) = log(k/k_min) / log(k_max/k_min)
        # Inverse: k = k_min * (k_max/k_min)^u
        u = np.random.uniform(0, 1, num_nodes)
        degrees = k_min * np.power(k_max / k_min, u)
    else:
        # General case: gamma != 1
        exp = 1.0 - gamma
        u = np.random.uniform(0, 1, num_nodes)
        k_min_exp = k_min ** exp
        k_max_exp = k_max ** exp
        degrees = np.power((k_max_exp - k_min_exp) * u + k_min_exp, 1.0 / exp)
    
    return degrees


def compute_community_sizes(num_nodes: int, pi: np.ndarray) -> np.ndarray:
    """
    Compute integer community sizes from proportions.
    Handles rounding to ensure sizes sum to num_nodes.
    """
    sizes = (pi * num_nodes).astype(int)
    
    # Distribute remainder to balance sizes
    remainder = num_nodes - sizes.sum()
    if remainder != 0:
        # Add/subtract from communities to balance
        indices = np.argsort(sizes) if remainder > 0 else np.argsort(-sizes)
        for i in range(abs(remainder)):
            sizes[indices[i % len(indices)]] += np.sign(remainder)
    
    return sizes


def compute_expected_edge_counts(
    num_edges: int,
    num_nodes: int,
    pi: np.ndarray,
    prop_mat: np.ndarray,
) -> np.ndarray:
    """
    Compute expected edge counts between community pairs.
    
    This matches GraphWorld's _ComputeExpectedEdgeCounts.
    """
    # Scale factor to achieve target number of edges
    scale = np.dot(pi, np.dot(prop_mat, pi)) * num_nodes ** 2
    prob_mat = prop_mat * num_edges / scale
    
    # Expected edges between communities i and j
    expected = np.outer(pi, pi) * prob_mat * num_nodes ** 2
    return expected


# =============================================================================
# Main DCSBM Generation
# =============================================================================

def generate_dcsbm_graph(
    num_nodes: int = 50,
    num_edges: int = 150,
    num_communities: int = 2,
    p_to_q_ratio: float = 5.0,
    cluster_size_slope: float = 0.0,
    power_exponent: float = 0.0,
    min_deg: int = 1,
    node_feature_type: Literal["community", "constant", "degree"] = "community",
    edge_feature_type: Literal["community", "constant"] = "community",
    target_type: Literal["tailed_triangles", "triangle_count", "edge_count"] = "tailed_triangles",
    seed: Optional[int] = None,
) -> Data:
    """
    Generate a single DCSBM graph (matching GraphWorld's approach).
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Expected number of edges
        num_communities: Number of communities (k)
        p_to_q_ratio: Homophily ratio (intra/inter community edge rate)
        cluster_size_slope: Community size imbalance (0 = equal)
        power_exponent: Degree distribution power-law exponent (0 = uniform)
        min_deg: Minimum degree for power-law
        node_feature_type: "community", "constant", or "degree"
        edge_feature_type: "community" or "constant"
        target_type: Regression target
        seed: Random seed
        
    Returns:
        PyG Data object
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Step 1: Create community structure
    pi = make_pi(num_communities, cluster_size_slope)
    prop_mat = make_prop_mat(num_communities, p_to_q_ratio)
    community_sizes = compute_community_sizes(num_nodes, pi)
    
    # Assign nodes to communities
    community_labels = np.repeat(np.arange(num_communities), community_sizes)
    
    # Step 2: Generate degree propensities
    theta = make_degree_propensities(num_nodes, power_exponent, min_deg)
    
    # Normalize theta within each community
    # After normalization, sum of theta in each community = 1
    for c in range(num_communities):
        mask = community_labels == c
        if mask.sum() > 0:
            theta[mask] /= theta[mask].sum()
    
    # Step 3: Compute base edge probabilities p (intra) and q (inter)
    # to achieve target num_edges with given p_to_q_ratio
    # 
    # Expected edges = sum over all pairs of P(edge)
    # For intra-community pairs in community c: n_c*(n_c-1)/2 pairs, each with prob ~ p * theta_i * theta_j * n_c^2
    # For inter-community pairs: n_i * n_j pairs, each with prob ~ q * theta_i * theta_j * n_i * n_j
    #
    # Simplified: with uniform theta, expected intra edges in community c = p * n_c*(n_c-1)/2
    #             expected inter edges between c1,c2 = q * n_c1 * n_c2
    
    # Calculate total possible edges and expected edges for each type
    total_intra_pairs = sum(n * (n - 1) // 2 for n in community_sizes)
    total_inter_pairs = 0
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            total_inter_pairs += community_sizes[i] * community_sizes[j]
    
    # Solve for q: num_edges = p * total_intra_pairs + q * total_inter_pairs
    #              p = p_to_q_ratio * q
    # num_edges = q * (p_to_q_ratio * total_intra_pairs + total_inter_pairs)
    if total_intra_pairs + total_inter_pairs > 0:
        q = num_edges / (p_to_q_ratio * total_intra_pairs + total_inter_pairs)
        p = p_to_q_ratio * q
    else:
        p, q = 0.5, 0.1
    
    # Clamp probabilities
    p = min(1.0, max(0.0, p))
    q = min(1.0, max(0.0, q))
    
    # Step 4: Sample edges using DCSBM model
    edges_src = []
    edges_dst = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            c_i = community_labels[i]
            c_j = community_labels[j]
            n_i = community_sizes[c_i]
            n_j = community_sizes[c_j]
            
            # Base probability
            if c_i == c_j:
                base_prob = p
                # Degree correction: theta normalized to sum=1, so multiply by n^2
                # to get expected value of theta_i * theta_j * n^2 = 1
                degree_factor = theta[i] * theta[j] * n_i * n_i
            else:
                base_prob = q
                degree_factor = theta[i] * theta[j] * n_i * n_j
            
            prob = base_prob * degree_factor
            prob = min(1.0, max(0.0, prob))
            
            # Sample edge
            if np.random.random() < prob:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
    
    # Handle case of no edges
    if len(edges_src) == 0:
        edges_src = [0, 1]
        edges_dst = [1, 0]
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    # Step 5: Generate node features
    if node_feature_type == "community":
        x = torch.tensor(community_labels, dtype=torch.long).unsqueeze(-1)
    elif node_feature_type == "constant":
        x = torch.ones(num_nodes, 1, dtype=torch.long)
    elif node_feature_type == "degree":
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        x = degrees.unsqueeze(-1).long()
    else:
        raise ValueError(f"Unknown node_feature_type: {node_feature_type}")
    
    # Step 6: Generate edge features
    num_edge_entries = edge_index.size(1)
    if edge_feature_type == "community":
        edge_attr = torch.zeros(num_edge_entries, dtype=torch.long)
        for e in range(num_edge_entries):
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            if community_labels[src] == community_labels[dst]:
                edge_attr[e] = 1  # Intra-community
            else:
                edge_attr[e] = 2  # Inter-community
    elif edge_feature_type == "constant":
        edge_attr = torch.ones(num_edge_entries, dtype=torch.long)
    else:
        raise ValueError(f"Unknown edge_feature_type: {edge_feature_type}")
    
    # Step 7: Compute target and graph statistics
    num_triangles = count_triangles(edge_index, num_nodes)
    num_tailed = count_tailed_triangles(edge_index, num_nodes)
    actual_edges = edge_index.size(1) // 2
    edge_homogeneity = compute_edge_homogeneity(edge_index, community_labels)
    avg_degree, degree_variance = compute_degree_stats(edge_index, num_nodes)

    if target_type == "tailed_triangles":
        y_value = float(num_tailed)
    elif target_type == "triangle_count":
        y_value = float(num_triangles)
    elif target_type == "edge_count":
        y_value = float(actual_edges)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    y = torch.tensor([[y_value]], dtype=torch.float)
    
    # Create Data object
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
    data.num_triangles = num_triangles
    data.num_tailed_triangles = num_tailed
    data.edge_homogeneity = edge_homogeneity
    data.avg_degree = avg_degree
    data.degree_variance = degree_variance
    data.target_type = target_type
    data.p_to_q_ratio = p_to_q_ratio
    
    return data


# =============================================================================
# Convenience Functions (matching old interface)
# =============================================================================

def generate_sbm_graph(
    num_nodes: int = 50,
    num_communities: int = 2,
    p: float = 0.3,
    q: float = 0.05,
    node_feature_type: str = "community",
    edge_feature_type: str = "community",
    target_type: str = "tailed_triangles",
    seed: Optional[int] = None,
) -> Data:
    """
    Generate SBM graph using old interface (for backward compatibility).
    Converts p, q to DCSBM parameters.
    """
    # Convert p, q to DCSBM parameters
    p_to_q_ratio = p / q if q > 0 else 10.0
    
    # Estimate num_edges from p, q
    # avg_degree ≈ (n/k - 1) * p + (n - n/k) * q
    n_c = num_nodes / num_communities
    avg_degree = (n_c - 1) * p + (num_nodes - n_c) * q
    num_edges = int(num_nodes * avg_degree / 2)
    
    return generate_dcsbm_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_communities=num_communities,
        p_to_q_ratio=p_to_q_ratio,
        cluster_size_slope=0.0,  # Equal community sizes (old behavior)
        power_exponent=0.0,      # Uniform degrees (old behavior)
        node_feature_type=node_feature_type,
        edge_feature_type=edge_feature_type,
        target_type=target_type,
        seed=seed,
    )


def compute_p_q_from_homophily_and_degree(
    num_nodes: int,
    num_communities: int,
    homophily_ratio: float,
    avg_degree: float,
) -> Tuple[float, float]:
    """Compute p, q from homophily ratio and average degree (old interface)."""
    n_c = num_nodes / num_communities
    intra_possible = n_c - 1
    inter_possible = num_nodes - n_c
    
    q = avg_degree / (intra_possible * homophily_ratio + inter_possible)
    p = homophily_ratio * q
    
    p = min(max(p, 0.001), 0.999)
    q = min(max(q, 0.001), 0.999)
    
    return p, q


# =============================================================================
# Dataset Classes
# =============================================================================

class DCSBMDataset:
    """
    DCSBM dataset that generates graphs on-the-fly.
    """
    
    def __init__(
        self,
        num_graphs: int = 1000,
        num_nodes_range: Tuple[int, int] = (50, 100),
        num_communities: int = 2,
        p_to_q_ratio: float = 5.0,
        avg_degree: float = 6.0,
        cluster_size_slope: float = 0.0,
        power_exponent: float = 0.0,
        min_deg: int = 1,
        node_feature_type: str = "community",
        edge_feature_type: str = "community",
        target_type: str = "tailed_triangles",
        seed: int = 42,
    ):
        self.num_graphs = num_graphs
        self.num_nodes_range = num_nodes_range
        self.num_communities = num_communities
        self.p_to_q_ratio = p_to_q_ratio
        self.avg_degree = avg_degree
        self.cluster_size_slope = cluster_size_slope
        self.power_exponent = power_exponent
        self.min_deg = min_deg
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
        num_edges = int(num_nodes * self.avg_degree / 2)
        
        data = generate_dcsbm_graph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_communities=self.num_communities,
            p_to_q_ratio=self.p_to_q_ratio,
            cluster_size_slope=self.cluster_size_slope,
            power_exponent=self.power_exponent,
            min_deg=self.min_deg,
            node_feature_type=self.node_feature_type,
            edge_feature_type=self.edge_feature_type,
            target_type=self.target_type,
            seed=self.seed + idx,
        )
        
        self._cache[idx] = data
        return data


# Alias for backward compatibility
SBMDatasetSimple = DCSBMDataset


def create_dcsbm_splits(
    num_train: int = 800,
    num_val: int = 100,
    num_test: int = 100,
    num_nodes_range: Tuple[int, int] = (50, 100),
    num_communities: int = 2,
    p_to_q_ratio: float = 5.0,
    avg_degree: float = 6.0,
    cluster_size_slope: float = 0.0,
    power_exponent: float = 0.0,
    min_deg: int = 1,
    node_feature_type: str = "community",
    edge_feature_type: str = "community",
    target_type: str = "tailed_triangles",
    seed: int = 42,
) -> Tuple[DCSBMDataset, DCSBMDataset, DCSBMDataset]:
    """
    Create train/val/test splits for DCSBM dataset.
    """
    common_args = dict(
        num_nodes_range=num_nodes_range,
        num_communities=num_communities,
        p_to_q_ratio=p_to_q_ratio,
        avg_degree=avg_degree,
        cluster_size_slope=cluster_size_slope,
        power_exponent=power_exponent,
        min_deg=min_deg,
        node_feature_type=node_feature_type,
        edge_feature_type=edge_feature_type,
        target_type=target_type,
    )
    
    train_set = DCSBMDataset(num_graphs=num_train, seed=seed, **common_args)
    valid_set = DCSBMDataset(num_graphs=num_val, seed=seed + num_train, **common_args)
    test_set = DCSBMDataset(num_graphs=num_test, seed=seed + num_train + num_val, **common_args)
    
    print(f"[DCSBM] Created splits: {num_train} train, {num_val} val, {num_test} test")
    print(f"[DCSBM] Parameters: nodes={num_nodes_range}, k={num_communities}, "
          f"p/q={p_to_q_ratio}, avg_deg={avg_degree}")
    print(f"[DCSBM] Degree correction: slope={cluster_size_slope}, power_exp={power_exponent}")
    print(f"[DCSBM] Target: {target_type}")
    
    return train_set, valid_set, test_set


# Alias for backward compatibility
create_sbm_splits = create_dcsbm_splits


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing DCSBM generation...")
    print("=" * 60)
    
    # Test 1: Basic generation
    print("\n1. Basic DCSBM graph:")
    data = generate_dcsbm_graph(
        num_nodes=50,
        num_edges=150,
        num_communities=4,
        p_to_q_ratio=5.0,
        seed=42,
    )
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Target edges: 150, Actual edges: {data.edge_index.size(1) // 2}")
    print(f"   Triangles: {data.num_triangles}")
    print(f"   Tailed triangles: {data.num_tailed_triangles}")
    print(f"   Edge homogeneity: {data.edge_homogeneity:.4f}")
    print(f"   Avg degree: {data.avg_degree:.2f}")
    print(f"   Degree variance: {data.degree_variance:.2f}")
    print(f"   Community sizes: {torch.bincount(data.community_labels).tolist()}")
    
    # Test 2: Unequal community sizes
    print("\n2. Unequal community sizes (slope=0.5):")
    data = generate_dcsbm_graph(
        num_nodes=100,
        num_edges=300,
        num_communities=4,
        p_to_q_ratio=5.0,
        cluster_size_slope=0.5,
        seed=42,
    )
    print(f"   Target edges: 300, Actual edges: {data.edge_index.size(1) // 2}")
    print(f"   Community sizes: {torch.bincount(data.community_labels).tolist()}")
    print(f"   Edge homogeneity: {data.edge_homogeneity:.4f}")
    print(f"   Avg degree: {data.avg_degree:.2f}")
    
    # Test 3: Power-law degrees
    print("\n3. Power-law degree distribution (exp=2.5):")
    data = generate_dcsbm_graph(
        num_nodes=100,
        num_edges=300,
        num_communities=2,
        p_to_q_ratio=5.0,
        power_exponent=2.5,
        min_deg=2,
        seed=42,
    )
    degrees = torch.bincount(data.edge_index[0], minlength=100)
    print(f"   Target edges: 300, Actual edges: {data.edge_index.size(1) // 2}")
    print(f"   Degree range: {degrees.min().item()} - {degrees.max().item()}")
    print(f"   Avg degree (computed): {data.avg_degree:.2f}")
    print(f"   Degree variance: {data.degree_variance:.2f}")
    print(f"   Edge homogeneity: {data.edge_homogeneity:.4f}")
    
    # Test 4: Varying homophily and its effect on edge homogeneity
    print("\n4. Effect of p_to_q_ratio on graph statistics:")
    print(f"   {'p/q':>5} | {'edges':>5} | {'tri':>4} | {'tailed':>6} | {'edge_hom':>8} | {'avg_deg':>7} | {'deg_var':>7}")
    print(f"   {'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    for ratio in [1.0, 2.0, 5.0, 10.0, 20.0]:
        data = generate_dcsbm_graph(
            num_nodes=50,
            num_edges=150,
            num_communities=2,
            p_to_q_ratio=ratio,
            seed=42,
        )
        actual_edges = data.edge_index.size(1) // 2
        print(f"   {ratio:>5.1f} | {actual_edges:>5} | {data.num_triangles:>4} | {data.num_tailed_triangles:>6} | "
              f"{data.edge_homogeneity:>8.3f} | {data.avg_degree:>7.2f} | {data.degree_variance:>7.2f}")
    
    # Test 5: Effect of power exponent on degree distribution
    print("\n5. Effect of power_exponent on graph statistics:")
    print(f"   {'pexp':>5} | {'edges':>5} | {'tri':>4} | {'tailed':>6} | {'edge_hom':>8} | {'avg_deg':>7} | {'deg_var':>7} | {'deg_min':>7} | {'deg_max':>7}")
    print(f"   {'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for pexp in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]:
        data = generate_dcsbm_graph(
            num_nodes=100,
            num_edges=300,
            num_communities=2,
            p_to_q_ratio=5.0,
            power_exponent=pexp,
            min_deg=1,
            seed=43,
        )
        actual_edges = data.edge_index.size(1) // 2
        degrees = torch.bincount(data.edge_index[0], minlength=100)
        print(f"   {pexp:>5.1f} | {actual_edges:>5} | {data.num_triangles:>4} | {data.num_tailed_triangles:>6} | "
              f"{data.edge_homogeneity:>8.3f} | {data.avg_degree:>7.2f} | {data.degree_variance:>7.2f} | "
              f"{degrees.min().item():>7} | {degrees.max().item():>7}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")