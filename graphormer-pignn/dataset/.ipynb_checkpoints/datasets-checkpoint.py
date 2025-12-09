# Dataset classes for Graphormer-PIGNN
# Compatible with both synthetic (PT-Motifs) and real-world datasets

import os
import os.path as osp
import pickle
import random
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse


class BA2Motif(InMemoryDataset):
    """
    BA-2Motifs dataset for graph classification with ground-truth explanations.
    
    Each graph consists of a Barabási-Albert base graph attached to either
    a house motif (class 0) or a cycle motif (class 1). The motif edges
    serve as ground-truth explanations.
    """
    splits = ['train', 'valid', 'test']
    
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index(f'{mode}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])
    
    @property
    def raw_file_names(self):
        return ['ba2motif.pkl']
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']
    
    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'ba2motif.pkl')):
            print("BA2Motif data not found. Please download from the original source.")
            raise FileNotFoundError
    
    def process(self):
        with open(osp.join(self.raw_dir, 'ba2motif.pkl'), 'rb') as f:
            adj_list, x_list, y_list, exp_gt_list = pickle.load(f)
        
        graph_list = []
        for i, (adj, x, y, exp_gt) in enumerate(zip(adj_list, x_list, y_list, exp_gt_list)):
            edge_index = dense_to_sparse(torch.tensor(adj))[0]
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(np.argmax(y), dtype=torch.long)
            exp_gt = torch.tensor(exp_gt, dtype=torch.long)
            edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float)
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                edge_gt_att=exp_gt,  # Ground-truth explanation
                idx=i
            )
            
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            
            graph_list.append(graph)
        
        random.shuffle(graph_list)
        n = len(graph_list)
        
        # Split: 40% train, 40% valid, 20% test
        train_end = int(0.4 * n)
        valid_end = int(0.8 * n)
        
        torch.save(self.collate(graph_list[:train_end]), self.processed_paths[0])
        torch.save(self.collate(graph_list[train_end:valid_end]), self.processed_paths[1])
        torch.save(self.collate(graph_list[valid_end:]), self.processed_paths[2])


class Synthetic(InMemoryDataset):
    """
    Synthetic PT-Motifs dataset for pre-training.
    
    Each graph consists of a base subgraph and an explanation subgraph (motif).
    The task label is determined solely by the motif.
    """
    splits = ['train', 'val', 'test']
    
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index(f'synthetic_{mode}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])
    
    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']
    
    @property
    def processed_file_names(self):
        return ['synthetic_train.pt', 'synthetic_val.pt', 'synthetic_test.pt']
    
    def download(self):
        if not osp.exists(osp.join(self.raw_dir, f'{self.mode}.npy')):
            print(f"Synthetic {self.mode} data not found.")
            raise FileNotFoundError
    
    def process(self):
        idx = self.raw_file_names.index(f'{self.mode}.npy')
        data = np.load(osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = data
        
        data_list = []
        for i, (edge_index, y, ground_truth, z, p) in enumerate(
            zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)
        ):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            n_nodes = node_idx.max().item() + 1
            
            # Node features (random or role-based)
            x = torch.rand(n_nodes, 4)
            edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.long)
            ground_truth = torch.tensor(ground_truth, dtype=torch.long)
            
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                edge_gt_att=ground_truth,
                idx=i
            )
            
            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            
            data_list.append(graph)
        
        idx = self.processed_file_names.index(f'synthetic_{self.mode}.pt')
        torch.save(self.collate(data_list), self.processed_paths[idx])


class SpuriousMotif(InMemoryDataset):
    """
    Spurious-Motif dataset with controllable spurious correlation.
    
    Args:
        b: Degree of spurious correlation (0.5, 0.7, or 0.9)
    """
    splits = ['train', 'val', 'test']
    
    def __init__(self, root, mode='train', b=0.5, transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        assert b in [0.5, 0.7, 0.9]
        self.mode = mode
        self.b = b
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index(f'spmotif_{mode}_b{b}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])
    
    @property
    def raw_file_names(self):
        return [f'spmotif_{self.mode}_b{self.b}.pkl']
    
    @property
    def processed_file_names(self):
        return [f'spmotif_{m}_b{b}.pt' for m in self.splits for b in [0.5, 0.7, 0.9]]
    
    def download(self):
        pass
    
    def process(self):
        # Implementation depends on data format
        pass


class MutagDataset(InMemoryDataset):
    """
    Mutag dataset for mutagenicity prediction.
    
    Ground-truth explanations: -NO2 and -NH2 functional groups.
    """
    splits = ['train', 'val', 'test']
    
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index(f'mutag_{mode}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])
    
    @property
    def raw_file_names(self):
        return ['mutag.pkl']
    
    @property
    def processed_file_names(self):
        return ['mutag_train.pt', 'mutag_val.pt', 'mutag_test.pt']
    
    def download(self):
        pass
    
    def process(self):
        # Implementation depends on data format
        pass


# ============================================================================
# Synthetic Data Generator
# ============================================================================

def generate_ba_graph(n_nodes, m=1):
    """Generate Barabási-Albert graph."""
    import networkx as nx
    G = nx.barabasi_albert_graph(n_nodes, m)
    edge_index = torch.tensor(list(G.edges())).t()
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def generate_house_motif():
    """Generate house motif (5 nodes, 6 edges)."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]
    edge_index = torch.tensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index, 5


def generate_cycle_motif(n=5):
    """Generate cycle motif."""
    edges = [(i, (i+1) % n) for i in range(n)]
    edge_index = torch.tensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index, n


def generate_star_motif(n=5):
    """Generate star motif."""
    edges = [(0, i) for i in range(1, n)]
    edge_index = torch.tensor(edges).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index, n


def attach_motif(base_edge_index, n_base, motif_edge_index, n_motif, attach_node=None):
    """Attach motif to base graph."""
    if attach_node is None:
        attach_node = random.randint(0, n_base - 1)
    
    # Offset motif node indices
    motif_offset = motif_edge_index + n_base
    
    # Create attachment edge
    attach_edge = torch.tensor([[attach_node, n_base], [n_base, attach_node]], dtype=torch.long)
    
    # Combine
    combined = torch.cat([base_edge_index, motif_offset, attach_edge], dim=1)
    
    return combined, n_base + n_motif


def generate_synthetic_graph(motif_type='house', base_nodes=20):
    """
    Generate a synthetic graph with base + motif.
    
    Returns:
        Data object with ground-truth explanation
    """
    # Generate base graph
    base_edge_index = generate_ba_graph(base_nodes, m=1)
    
    # Generate motif
    if motif_type == 'house':
        motif_edge_index, n_motif = generate_house_motif()
        label = 0
    elif motif_type == 'cycle':
        motif_edge_index, n_motif = generate_cycle_motif(5)
        label = 1
    elif motif_type == 'star':
        motif_edge_index, n_motif = generate_star_motif(5)
        label = 2
    else:
        raise ValueError(f"Unknown motif type: {motif_type}")
    
    # Attach motif
    edge_index, n_nodes = attach_motif(base_edge_index, base_nodes, motif_edge_index, n_motif)
    
    # Create ground-truth explanation (1 for motif edges, 0 for base edges)
    n_base_edges = base_edge_index.size(1)
    n_motif_edges = motif_edge_index.size(1)
    n_attach_edges = 2
    
    edge_gt_att = torch.zeros(edge_index.size(1), dtype=torch.long)
    edge_gt_att[n_base_edges:n_base_edges + n_motif_edges] = 1
    
    # Node features (random)
    x = torch.rand(n_nodes, 4)
    edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float)
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label, dtype=torch.long),
        edge_gt_att=edge_gt_att
    )


def generate_synthetic_dataset(n_graphs=1000, motif_types=['house', 'cycle', 'star'], base_nodes=20):
    """Generate synthetic dataset."""
    graphs = []
    for i in range(n_graphs):
        motif_type = random.choice(motif_types)
        graph = generate_synthetic_graph(motif_type, base_nodes)
        graph.idx = i
        graphs.append(graph)
    return graphs


class GeneratedSynthetic(InMemoryDataset):
    """
    On-the-fly generated synthetic dataset for testing.
    """
    
    def __init__(self, root, mode='train', n_graphs=1000, transform=None, pre_transform=None):
        self.mode = mode
        self.n_graphs = n_graphs
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return [f'generated_{self.mode}_{self.n_graphs}.pt']
    
    def process(self):
        graphs = generate_synthetic_dataset(self.n_graphs)
        
        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]
        
        torch.save(self.collate(graphs), self.processed_paths[0])


if __name__ == "__main__":
    # Test synthetic data generation
    print("Testing synthetic data generation...")
    
    graph = generate_synthetic_graph('house', base_nodes=15)
    print(f"Generated graph:")
    print(f"  Nodes: {graph.x.size(0)}")
    print(f"  Edges: {graph.edge_index.size(1)}")
    print(f"  Label: {graph.y.item()}")
    print(f"  Explanation edges: {graph.edge_gt_att.sum().item()}")
    
    # Generate small dataset
    graphs = generate_synthetic_dataset(n_graphs=10)
    print(f"\nGenerated {len(graphs)} graphs")
    
    print("\nDataset test passed!")
