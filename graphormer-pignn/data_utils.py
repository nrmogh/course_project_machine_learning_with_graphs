# Data preprocessing utilities for Graphormer-PIGNN
# Converts PyTorch Geometric data to Graphormer format

import torch
import numpy as np
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional
import warnings


def floyd_warshall_python(adj_matrix):
    """
    Pure Python implementation of Floyd-Warshall algorithm.
    Computes shortest path distances and predecessor matrix.
    
    Args:
        adj_matrix: numpy array [N, N] boolean adjacency matrix
    Returns:
        dist: [N, N] shortest path distances
        path: [N, N] predecessor matrix for path reconstruction
    """
    n = adj_matrix.shape[0]
    
    # Initialize distance matrix
    dist = np.full((n, n), 510, dtype=np.int64)  # 510 = unreachable
    path = np.full((n, n), -1, dtype=np.int64)
    
    # Set distances for existing edges
    for i in range(n):
        dist[i, i] = 0
        for j in range(n):
            if adj_matrix[i, j]:
                dist[i, j] = 1
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    path[i, j] = k
    
    return dist, path


def get_path_edges(path, i, j):
    """Recursively get all intermediate nodes on shortest path."""
    k = path[i, j]
    if k == -1:
        return []
    return get_path_edges(path, i, k) + [k] + get_path_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):
    """
    Generate edge input features along shortest paths.
    
    Args:
        max_dist: maximum path length
        path: [N, N] predecessor matrix
        edge_feat: [N, N, F] edge features
    Returns:
        edge_input: [N, N, max_dist, F] edge features along paths
    """
    n = path.shape[0]
    f = edge_feat.shape[-1]
    edge_input = np.full((n, n, max_dist, f), -1, dtype=np.int64)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path[i, j] == 510:  # unreachable
                continue
            
            # Reconstruct path
            full_path = [i] + get_path_edges(path, i, j) + [j]
            num_edges = len(full_path) - 1
            
            for k in range(min(num_edges, max_dist)):
                edge_input[i, j, k, :] = edge_feat[full_path[k], full_path[k+1], :]
    
    return edge_input


def convert_to_single_emb(x, offset=64, max_value=511):
    """
    Convert multi-feature node attributes to single embedding indices.
    
    Args:
        x: Input tensor of indices
        offset: Offset between features (default 64)
        max_value: Maximum allowed index value (default 511 for num_atoms=512)
    Returns:
        Tensor with combined embedding indices, clamped to max_value
    """
    if len(x.size()) == 1:
        x = x.unsqueeze(-1)
    feature_num = x.size(1)
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long, device=x.device)
    x = x + feature_offset
    # Clamp to valid embedding range
    x = x.clamp(0, max_value)
    return x


def preprocess_pyg_item(item: Data, max_dist: int = 20) -> Data:
    """
    Preprocess a single PyG Data object to add Graphormer-required features.
    
    Args:
        item: PyG Data object with x, edge_index, edge_attr, y
        max_dist: maximum distance for edge encoding along paths
    Returns:
        item: augmented with attn_bias, spatial_pos, attn_edge_type, 
              in_degree, out_degree, edge_input
    """
    edge_index = item.edge_index
    x = item.x
    N = x.size(0)
    
    # Handle edge attributes
    if hasattr(item, 'edge_attr') and item.edge_attr is not None:
        edge_attr = item.edge_attr
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr.unsqueeze(-1)
    else:
        edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.long)
    
    # Convert node features to embedding indices
    if x.dtype == torch.float:
        # Discretize continuous features (simple binning to 0-31 range)
        # This keeps indices small when combined with feature offsets
        x_discrete = (x * 8).long().clamp(0, 31)
    else:
        x_discrete = x.long().clamp(0, 31)
    x_emb = convert_to_single_emb(x_discrete, offset=64, max_value=511)
    
    # Build adjacency matrix
    adj = torch.zeros(N, N, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    
    # Edge type matrix
    n_edge_features = edge_attr.size(-1)
    attn_edge_type = torch.zeros(N, N, n_edge_features, dtype=torch.long)
    edge_attr_emb = convert_to_single_emb(edge_attr.long()) + 1
    attn_edge_type[edge_index[0], edge_index[1]] = edge_attr_emb
    
    # Compute shortest paths
    adj_np = adj.numpy()
    dist, path = floyd_warshall_python(adj_np)
    
    # Spatial position (shortest path distance)
    spatial_pos = torch.from_numpy(dist).long()
    
    # Edge input along shortest paths
    actual_max_dist = min(max_dist, np.max(dist[dist < 510]) if np.any(dist < 510) else 1)
    actual_max_dist = max(actual_max_dist, 1)
    edge_input = gen_edge_input(actual_max_dist, path, attn_edge_type.numpy())
    edge_input = torch.from_numpy(edge_input).long()
    
    # Attention bias (initialized to zeros, with graph token)
    attn_bias = torch.zeros(N + 1, N + 1, dtype=torch.float)
    
    # Degree
    in_degree = adj.long().sum(dim=1)
    out_degree = adj.long().sum(dim=0)
    
    # Store augmented features
    item.x_graphormer = x_emb
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = in_degree
    item.out_degree = out_degree
    item.edge_input = edge_input
    
    return item


def pad_1d(x, padlen, pad_value=0):
    """Pad 1D tensor."""
    x = x + 1  # Shift for padding_idx=0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full([padlen], pad_value, dtype=x.dtype, device=x.device)
        new_x[:xlen] = x
        x = new_x
    return x


def pad_2d(x, padlen, pad_value=0):
    """Pad 2D tensor along first dimension."""
    x = x + 1  # Shift for padding_idx=0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full([padlen, x.size(1)], pad_value, dtype=x.dtype, device=x.device)
        new_x[:xlen, :] = x
        x = new_x
    return x


def pad_attn_bias(x, padlen):
    """Pad attention bias matrix."""
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full([padlen, padlen], float('-inf'), dtype=x.dtype, device=x.device)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0  # New nodes can attend to old nodes
        x = new_x
    return x


def pad_spatial_pos(x, padlen):
    """Pad spatial position matrix."""
    x = x + 1  # Shift for padding_idx=0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.zeros([padlen, padlen], dtype=x.dtype, device=x.device)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x


def pad_edge_type(x, padlen):
    """Pad edge type tensor."""
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.zeros([padlen, padlen, x.size(-1)], dtype=x.dtype, device=x.device)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x


def pad_3d(x, padlen1, padlen2, padlen3):
    """Pad 3D tensor for edge input."""
    x = x + 1  # Shift for padding_idx=0
    xlen1, xlen2, xlen3 = x.size()[:3]
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = torch.zeros([padlen1, padlen2, padlen3, x.size(-1)], dtype=x.dtype, device=x.device)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x


class GraphormerCollator:
    """
    Collator for batching PyG data into Graphormer format.
    
    Handles:
    1. Padding all graphs to max size in batch
    2. Creating batched tensors for Graphormer
    3. Preserving PyG batch for edge indexing
    """
    
    def __init__(self, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
    
    def __call__(self, items: List[Data]) -> Dict[str, Any]:
        """
        Collate list of PyG Data objects.
        
        Returns:
            dict with:
                - graphormer_data: dict for Graphormer model
                - pyg_batch: PyG Batch object for GNN predictor
        """
        # Filter out items that are too large
        items = [item for item in items if item.x.size(0) <= self.max_node]
        
        if len(items) == 0:
            raise ValueError("All items filtered out due to size")
        
        # Preprocess items that haven't been preprocessed
        processed_items = []
        for item in items:
            if not hasattr(item, 'attn_bias'):
                item = preprocess_pyg_item(item, self.multi_hop_max_dist)
            processed_items.append(item)
        
        # Create clean copies for PyG batching (exclude Graphormer-specific tensors)
        # These tensors have shape [N, N] which can't be batched by PyG
        graphormer_keys = {'attn_bias', 'attn_edge_type', 'spatial_pos', 'in_degree', 
                          'out_degree', 'edge_input', 'x_graphormer'}
        
        clean_items = []
        for item in processed_items:
            clean_data = Data()
            for key in item.keys():
                if key not in graphormer_keys:
                    setattr(clean_data, key, getattr(item, key))
            clean_items.append(clean_data)
        
        # Create PyG batch (needed for edge indices and batch assignment)
        pyg_batch = Batch.from_data_list(clean_items)
        
        # Collect Graphormer features
        xs = [item.x_graphormer if hasattr(item, 'x_graphormer') 
              else convert_to_single_emb(item.x.long().clamp(0, 31), offset=64, max_value=511) for item in processed_items]
        attn_biases = [item.attn_bias for item in processed_items]
        attn_edge_types = [item.attn_edge_type for item in processed_items]
        spatial_poses = [item.spatial_pos for item in processed_items]
        in_degrees = [item.in_degree for item in processed_items]
        out_degrees = [item.out_degree for item in processed_items]
        edge_inputs = [item.edge_input[:, :, :self.multi_hop_max_dist, :] for item in processed_items]
        ys = [item.y for item in processed_items]
        
        # Apply spatial position cutoff
        for idx in range(len(attn_biases)):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= self.spatial_pos_max] = float('-inf')
        
        # Find max sizes
        max_node_num = max(x.size(0) for x in xs)
        max_dist = max(e.size(-2) for e in edge_inputs)
        
        # Pad and stack
        x = torch.stack([pad_2d(x, max_node_num) for x in xs])
        attn_bias = torch.stack([pad_attn_bias(ab, max_node_num + 1) for ab in attn_biases])
        attn_edge_type = torch.stack([pad_edge_type(et, max_node_num) for et in attn_edge_types])
        spatial_pos = torch.stack([pad_spatial_pos(sp, max_node_num) for sp in spatial_poses])
        in_degree = torch.stack([pad_1d(d, max_node_num) for d in in_degrees])
        out_degree = torch.stack([pad_1d(d, max_node_num) for d in out_degrees])
        edge_input = torch.stack([pad_3d(e, max_node_num, max_node_num, max_dist) for e in edge_inputs])
        
        # Handle y tensor
        if ys[0].dim() == 0:
            y = torch.stack(ys)
        else:
            y = torch.cat(ys)
        
        # Collect ground truth explanations if available
        edge_gt_att = None
        if hasattr(items[0], 'edge_gt_att'):
            edge_gt_att = pyg_batch.edge_gt_att
        elif hasattr(items[0], 'exp_gt'):
            edge_gt_att = pyg_batch.exp_gt
        
        graphormer_data = {
            'x': x,
            'attn_bias': attn_bias,
            'attn_edge_type': attn_edge_type,
            'spatial_pos': spatial_pos,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'edge_input': edge_input,
            'y': y,
        }
        
        return {
            'graphormer_data': graphormer_data,
            'pyg_batch': pyg_batch,
            'edge_gt_att': edge_gt_att,
        }


def create_dataloader(dataset, batch_size, shuffle=True, **kwargs):
    """Create DataLoader with Graphormer collator."""
    from torch.utils.data import DataLoader
    
    collator = GraphormerCollator(**kwargs)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collator
    )


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing data preprocessing...")
    
    # Create a simple test graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 3)
    edge_attr = torch.ones(6, 1)
    y = torch.tensor([0])
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Preprocess
    processed = preprocess_pyg_item(data)
    print(f"Original nodes: {x.size(0)}")
    print(f"Spatial pos shape: {processed.spatial_pos.shape}")
    print(f"Attn bias shape: {processed.attn_bias.shape}")
    print(f"In degree: {processed.in_degree}")
    
    # Test collator
    collator = GraphormerCollator()
    batch = collator([data, data])
    print(f"\nBatch x shape: {batch['graphormer_data']['x'].shape}")
    print(f"PyG batch edge_index shape: {batch['pyg_batch'].edge_index.shape}")
    
    print("\nData preprocessing test passed!")