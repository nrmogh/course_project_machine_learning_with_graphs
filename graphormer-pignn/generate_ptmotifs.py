#!/usr/bin/env python
"""
Generate PT-Motifs dataset using the original π-GNN data generator
and convert it to the format compatible with Graphormer-PIGNN.

This script:
1. Uses the pi-gnn data-generator code to create synthetic graphs
2. Saves them in .npy format (same as original pi-gnn)
3. Provides a dataset class that loads them for Graphormer-PIGNN

Usage:
    python generate_ptmotifs.py --output_dir data/PT-Motifs --n_train 20000 --n_val 5000 --n_test 5000

Or import and use programmatically:
    from generate_ptmotifs import generate_ptmotifs_dataset, PTMotifsDataset
"""

import os
import sys
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data


# ============================================================================
# Graph Generation Functions (adapted from pi-gnn/data-generator)
# ============================================================================

def perturb(graph_list, p, id=None):
    """Perturb graphs by adding edges."""
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            if (id is None) or (id[u] == 0 or id[v] == 0):
                G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list


def find_ground_truth(edge_index, role_ids):
    """Find ground-truth edge labels (1 if both endpoints are in motif)."""
    row, col = edge_index
    gd = np.array(role_ids[row] > 0, dtype=np.float64) * \
         np.array(role_ids[col] > 0, dtype=np.float64)
    return gd


def ba_graph(n, m=2):
    """Generate Barabási-Albert graph."""
    return nx.barabasi_albert_graph(n, m)


def tree_graph(height):
    """Generate balanced tree."""
    return nx.balanced_tree(2, height)


def ladder_graph(n):
    """Generate ladder graph."""
    return nx.ladder_graph(n)


def wheel_graph(n):
    """Generate wheel graph."""
    return nx.wheel_graph(n)


def clique_graph(n):
    """Generate complete graph (clique)."""
    return nx.complete_graph(n)


def get_base_graph(base_type, width):
    """Get base graph of specified type."""
    if base_type == 'ba':
        return ba_graph(width, m=2)
    elif base_type == 'tree':
        return tree_graph(width)
    elif base_type == 'ladder':
        return ladder_graph(width)
    elif base_type == 'wheel':
        return wheel_graph(width)
    elif base_type == 'clique':
        return clique_graph(width)
    else:
        raise ValueError(f"Unknown base type: {base_type}")


def house_motif():
    """Generate house motif (5 nodes)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)])
    return G, 5


def cycle_motif(n=6):
    """Generate cycle motif."""
    G = nx.cycle_graph(n)
    return G, n


def diamond_motif():
    """Generate diamond motif (4 nodes)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    return G, 4


def crane_motif():
    """Generate crane/fan motif (6 nodes)."""
    G = nx.Graph()
    # Fan-like structure
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (4, 5), (3, 5)])
    return G, 6


def star_motif(n=5):
    """Generate star motif."""
    G = nx.star_graph(n - 1)
    return G, n


def attach_motif(base_G, motif_G, motif_size):
    """Attach motif to base graph."""
    # Get number of nodes in base
    n_base = base_G.number_of_nodes()
    
    # Create combined graph
    G = nx.Graph()
    G.add_nodes_from(range(n_base + motif_size))
    G.add_edges_from(base_G.edges())
    
    # Add motif edges (offset by n_base)
    for u, v in motif_G.edges():
        G.add_edge(u + n_base, v + n_base)
    
    # Connect motif to base (attach first motif node to random base node)
    attach_node = np.random.randint(0, n_base)
    G.add_edge(attach_node, n_base)
    
    # Create role_id: 0 for base nodes, 1+ for motif nodes
    role_id = np.zeros(n_base + motif_size, dtype=np.int32)
    role_id[n_base:] = 1
    
    return G, role_id


def generate_single_graph(motif_type, base_type='ba', base_width=20):
    """Generate a single graph with base + motif."""
    
    # Get base graph
    base_G = get_base_graph(base_type, base_width)
    
    # Get motif
    if motif_type == 'house':
        motif_G, motif_size = house_motif()
        label = 0
    elif motif_type == 'cycle':
        motif_G, motif_size = cycle_motif()
        label = 1
    elif motif_type == 'crane':
        motif_G, motif_size = crane_motif()
        label = 2
    elif motif_type == 'diamond':
        motif_G, motif_size = diamond_motif()
        label = 3
    elif motif_type == 'star':
        motif_G, motif_size = star_motif()
        label = 4
    else:
        raise ValueError(f"Unknown motif type: {motif_type}")
    
    # Attach motif to base
    G, role_id = attach_motif(base_G, motif_G, motif_size)
    
    # Optional perturbation
    G = perturb([G], 0.0, id=role_id)[0]
    
    # Get edge index and ground truth
    edge_index = np.array(G.edges(), dtype=np.int32).T
    ground_truth = find_ground_truth(edge_index, role_id)
    
    # Get positions for visualization
    pos = np.array(list(nx.spring_layout(G).values()))
    
    return edge_index, label, ground_truth, role_id, pos


def graph_stats(base_num):
    """Get base type and width for normal-sized graphs."""
    configs = {
        1: ('tree', np.random.randint(2, 4)),
        2: ('ladder', np.random.randint(8, 12)),
        3: ('wheel', np.random.randint(15, 20)),
        4: ('clique', np.random.randint(15, 20)),
        5: ('ba', np.random.randint(20, 25)),
    }
    return configs.get(base_num, ('ba', 20))


def graph_stats_large(base_num):
    """Get base type and width for large graphs."""
    configs = {
        1: ('tree', np.random.randint(3, 6)),
        2: ('ladder', np.random.randint(30, 50)),
        3: ('wheel', np.random.randint(60, 80)),
        4: ('clique', np.random.randint(60, 80)),
        5: ('ba', np.random.randint(60, 80)),
    }
    return configs.get(base_num, ('ba', 60))


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_ptmotifs_dataset(
    n_train=20000, 
    n_val=5000, 
    n_test=5000,
    motif_types=['cycle', 'house', 'crane', 'diamond', 'star'],
    include_large=True,
    output_dir='data/PT-Motifs/raw',
    seed=42
):
    """
    Generate PT-Motifs dataset similar to the original π-GNN paper.
    
    Args:
        n_train: Number of training graphs
        n_val: Number of validation graphs  
        n_test: Number of test graphs
        motif_types: List of motif types to include
        include_large: Whether to include large graphs
        output_dir: Directory to save the .npy files
        seed: Random seed
    
    Returns:
        Paths to the generated files
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    def generate_split(n_graphs, split_name):
        edge_index_list = []
        label_list = []
        ground_truth_list = []
        role_id_list = []
        pos_list = []
        
        graphs_per_motif = n_graphs // len(motif_types)
        
        for motif_type in motif_types:
            print(f"  Generating {graphs_per_motif} graphs with {motif_type} motif...")
            
            for _ in tqdm(range(graphs_per_motif // 2)):
                # Normal-sized graphs
                base_num = np.random.choice([1, 2, 3, 4, 5])
                base_type, base_width = graph_stats(base_num)
                
                edge_index, label, ground_truth, role_id, pos = generate_single_graph(
                    motif_type, base_type, base_width
                )
                
                edge_index_list.append(edge_index)
                label_list.append(label)
                ground_truth_list.append(ground_truth)
                role_id_list.append(role_id)
                pos_list.append(pos)
            
            if include_large:
                for _ in tqdm(range(graphs_per_motif // 2)):
                    # Large graphs
                    base_num = np.random.choice([1, 2, 3, 4, 5])
                    base_type, base_width = graph_stats_large(base_num)
                    
                    edge_index, label, ground_truth, role_id, pos = generate_single_graph(
                        motif_type, base_type, base_width
                    )
                    
                    edge_index_list.append(edge_index)
                    label_list.append(label)
                    ground_truth_list.append(ground_truth)
                    role_id_list.append(role_id)
                    pos_list.append(pos)
        
        # Shuffle
        indices = np.random.permutation(len(edge_index_list))
        edge_index_list = [edge_index_list[i] for i in indices]
        label_list = [label_list[i] for i in indices]
        ground_truth_list = [ground_truth_list[i] for i in indices]
        role_id_list = [role_id_list[i] for i in indices]
        pos_list = [pos_list[i] for i in indices]
        
        # Save - use object array to handle variable-sized graphs
        save_path = os.path.join(output_dir, f'{split_name}.npy')
        
        # Create object array to handle inhomogeneous shapes
        data_to_save = np.array(
            [edge_index_list, label_list, ground_truth_list, role_id_list, pos_list],
            dtype=object
        )
        np.save(save_path, data_to_save, allow_pickle=True)
        print(f"  Saved {len(edge_index_list)} graphs to {save_path}")
        
        return save_path
    
    print("Generating training set...")
    train_path = generate_split(n_train, 'train')
    
    print("Generating validation set...")
    val_path = generate_split(n_val, 'val')
    
    print("Generating test set...")
    test_path = generate_split(n_test, 'test')
    
    return train_path, val_path, test_path


# ============================================================================
# PyTorch Geometric Dataset Class
# ============================================================================

class PTMotifsDataset(InMemoryDataset):
    """
    PT-Motifs dataset for Graphormer-PIGNN.
    
    Compatible with the original π-GNN .npy format.
    """
    splits = ['train', 'val', 'test']
    
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index(f'ptmotifs_{mode}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])
    
    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']
    
    @property
    def processed_file_names(self):
        return ['ptmotifs_train.pt', 'ptmotifs_val.pt', 'ptmotifs_test.pt']
    
    def download(self):
        # Check if raw files exist
        for fname in self.raw_file_names:
            if not os.path.exists(os.path.join(self.raw_dir, fname)):
                print(f"Raw file {fname} not found in {self.raw_dir}")
                print("Please generate the dataset first using:")
                print(f"  python generate_ptmotifs.py --output_dir {self.raw_dir}")
                raise FileNotFoundError(f"Missing {fname}")
    
    def process(self):
        for mode in self.splits:
            idx = self.raw_file_names.index(f'{mode}.npy')
            raw_path = os.path.join(self.raw_dir, self.raw_file_names[idx])
            
            print(f"Processing {mode} split from {raw_path}...")
            
            edge_index_list, label_list, ground_truth_list, role_id_list, pos_list = \
                np.load(raw_path, allow_pickle=True)
            
            data_list = []
            for i, (edge_index, y, ground_truth, role_id, pos) in enumerate(
                zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos_list)
            ):
                edge_index = torch.from_numpy(edge_index).long()
                
                # Get number of nodes
                n_nodes = edge_index.max().item() + 1
                
                # Node features (random or one-hot based on role)
                x = torch.rand(n_nodes, 4)
                
                # Edge attributes
                edge_attr = torch.ones(edge_index.size(1), 1)
                
                # Label
                y = torch.tensor(y, dtype=torch.long)
                
                # Ground-truth edge explanation
                edge_gt_att = torch.from_numpy(ground_truth).long()
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    edge_gt_att=edge_gt_att,
                    idx=i
                )
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                data_list.append(data)
            
            proc_idx = self.processed_file_names.index(f'ptmotifs_{mode}.pt')
            torch.save(self.collate(data_list), self.processed_paths[proc_idx])
            print(f"  Saved {len(data_list)} graphs to {self.processed_paths[proc_idx]}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PT-Motifs dataset')
    parser.add_argument('--output_dir', type=str, default='data/PT-Motifs/raw',
                       help='Output directory for raw .npy files')
    parser.add_argument('--n_train', type=int, default=20000, help='Number of training graphs')
    parser.add_argument('--n_val', type=int, default=5000, help='Number of validation graphs')
    parser.add_argument('--n_test', type=int, default=5000, help='Number of test graphs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--motifs', type=str, default='cycle,house,crane,diamond,star',
                       help='Comma-separated list of motif types')
    parser.add_argument('--no_large', action='store_true', help='Exclude large graphs')
    
    args = parser.parse_args()
    
    motif_types = args.motifs.split(',')
    
    print("=" * 60)
    print("PT-Motifs Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Training graphs: {args.n_train}")
    print(f"Validation graphs: {args.n_val}")
    print(f"Test graphs: {args.n_test}")
    print(f"Motif types: {motif_types}")
    print(f"Include large graphs: {not args.no_large}")
    print("=" * 60)
    
    generate_ptmotifs_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        motif_types=motif_types,
        include_large=not args.no_large,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\nDone! You can now use the dataset with:")
    print(f"  from generate_ptmotifs import PTMotifsDataset")
    print(f"  dataset = PTMotifsDataset('data/PT-Motifs', mode='train')")