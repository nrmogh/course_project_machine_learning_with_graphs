#!/usr/bin/env python
"""
Test script for Graphormer-PIGNN

This script verifies that all components work correctly:
1. Model creation and forward pass
2. Data preprocessing
3. Synthetic data generation
4. Training loop (one epoch)

Run with: python test_integration.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

print("=" * 60)
print("Graphormer-PIGNN Integration Test")
print("=" * 60)

# ============================================================================
# Test 1: Model Creation
# ============================================================================
print("\n[Test 1] Model Creation...")

from model import GraphormerPIGNN, count_parameters

model = GraphormerPIGNN(
    num_atoms=64,
    num_in_degree=32,
    num_out_degree=32,
    num_edges=64,
    num_spatial=32,
    num_encoder_layers=2,  # Small for testing
    embedding_dim=64,
    ffn_embedding_dim=64,
    num_attention_heads=4,
    dropout=0.1,
    attention_dropout=0.1,
    edge_hidden_dim=32,
    predictor_hidden_dim=32,
    num_classes=3,
    predictor_layers=2,
)

n_params = count_parameters(model)
print(f"  Model created with {n_params:,} parameters")
assert n_params > 0, "Model has no parameters!"
print("  ✓ Model creation passed")

# ============================================================================
# Test 2: Data Preprocessing
# ============================================================================
print("\n[Test 2] Data Preprocessing...")

from data_utils import preprocess_pyg_item, GraphormerCollator

# Create a simple test graph
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], 
                           [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
x = torch.randn(4, 3)
edge_attr = torch.ones(8, 1)
y = torch.tensor([0])
edge_gt_att = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])  # Some edges are explanatory

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_gt_att=edge_gt_att)

# Preprocess
processed = preprocess_pyg_item(data)

assert hasattr(processed, 'spatial_pos'), "Missing spatial_pos"
assert hasattr(processed, 'attn_bias'), "Missing attn_bias"
assert hasattr(processed, 'in_degree'), "Missing in_degree"
assert processed.spatial_pos.shape == (4, 4), f"Wrong spatial_pos shape: {processed.spatial_pos.shape}"

print(f"  Spatial positions computed: {processed.spatial_pos.shape}")
print(f"  In-degree: {processed.in_degree.tolist()}")
print("  ✓ Data preprocessing passed")

# ============================================================================
# Test 3: Collator (Batching)
# ============================================================================
print("\n[Test 3] Collator (Batching)...")

collator = GraphormerCollator(max_node=64, multi_hop_max_dist=5, spatial_pos_max=20)

# Create batch of 3 graphs
graphs = [data, data, data]
batch = collator(graphs)

graphormer_data = batch['graphormer_data']
pyg_batch = batch['pyg_batch']

print(f"  Batch x shape: {graphormer_data['x'].shape}")
print(f"  Batch attn_bias shape: {graphormer_data['attn_bias'].shape}")
print(f"  PyG edge_index shape: {pyg_batch.edge_index.shape}")
print(f"  Edge GT labels: {batch['edge_gt_att'].shape}")

assert graphormer_data['x'].shape[0] == 3, "Wrong batch size"
assert pyg_batch.batch.max().item() == 2, "Wrong PyG batch"
print("  ✓ Collator passed")

# ============================================================================
# Test 4: Forward Pass
# ============================================================================
print("\n[Test 4] Forward Pass...")

# Move to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

model = model.to(device)
graphormer_data = {k: v.to(device) for k, v in graphormer_data.items()}
pyg_batch = pyg_batch.to(device)

# Forward pass
model.eval()
with torch.no_grad():
    edge_scores, logits = model(graphormer_data, pyg_batch)

print(f"  Edge scores shape: {edge_scores.shape}")
print(f"  Edge scores range: [{edge_scores.min():.3f}, {edge_scores.max():.3f}]")
print(f"  Logits shape: {logits.shape}")

assert edge_scores.shape[0] == pyg_batch.edge_index.shape[1], \
    f"Edge scores ({edge_scores.shape[0]}) != edges ({pyg_batch.edge_index.shape[1]})"
assert logits.shape == (3, 3), f"Wrong logits shape: {logits.shape}"
assert (edge_scores >= 0).all() and (edge_scores <= 1).all(), "Edge scores not in [0,1]"
print("  ✓ Forward pass passed")

# ============================================================================
# Test 5: Loss Computation
# ============================================================================
print("\n[Test 5] Loss Computation...")

edge_gt_att = batch['edge_gt_att'].to(device).float()

exp_loss = F.binary_cross_entropy(edge_scores, edge_gt_att)
pred_loss = F.cross_entropy(logits, pyg_batch.y)
total_loss = exp_loss + pred_loss

print(f"  Explanation loss (BCE): {exp_loss.item():.4f}")
print(f"  Prediction loss (CE): {pred_loss.item():.4f}")
print(f"  Total loss: {total_loss.item():.4f}")

assert not torch.isnan(total_loss), "Loss is NaN!"
print("  ✓ Loss computation passed")

# ============================================================================
# Test 6: Backward Pass
# ============================================================================
print("\n[Test 6] Backward Pass...")

model.train()
edge_scores, logits = model(graphormer_data, pyg_batch)
exp_loss = F.binary_cross_entropy(edge_scores, edge_gt_att)
pred_loss = F.cross_entropy(logits, pyg_batch.y)
total_loss = exp_loss + pred_loss

total_loss.backward()

# Check gradients exist
has_grad = False
for name, param in model.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        has_grad = True
        break

assert has_grad, "No gradients computed!"
print("  ✓ Backward pass passed (gradients computed)")

# ============================================================================
# Test 7: Synthetic Data Generation
# ============================================================================
print("\n[Test 7] Synthetic Data Generation...")

from dataset.datasets import generate_synthetic_dataset

graphs = generate_synthetic_dataset(n_graphs=10, motif_types=['house', 'cycle', 'star'])

print(f"  Generated {len(graphs)} graphs")
for i, g in enumerate(graphs[:3]):
    print(f"    Graph {i}: {g.x.shape[0]} nodes, {g.edge_index.shape[1]} edges, "
          f"label={g.y.item()}, explanation edges={g.edge_gt_att.sum().item()}")

assert len(graphs) == 10, "Wrong number of graphs"
assert all(hasattr(g, 'edge_gt_att') for g in graphs), "Missing ground-truth explanations"
print("  ✓ Synthetic data generation passed")

# ============================================================================
# Test 8: Evaluation Metrics
# ============================================================================
print("\n[Test 8] Evaluation Metrics...")

from utils.evaluation import compute_explanation_metrics, compute_prediction_metrics

# Use model predictions
model.eval()
with torch.no_grad():
    edge_scores, logits = model(graphormer_data, pyg_batch)

exp_metrics = compute_explanation_metrics(edge_scores.cpu(), edge_gt_att.cpu())
pred_metrics = compute_prediction_metrics(logits.cpu(), pyg_batch.y.cpu())

print(f"  Explanation ROC-AUC: {exp_metrics['roc_auc']:.4f}")
print(f"  Explanation Accuracy: {exp_metrics['accuracy']:.4f}")
print(f"  Prediction Accuracy: {pred_metrics['accuracy']:.4f}")
print("  ✓ Evaluation metrics passed")

# ============================================================================
# Test 9: Mini Training Loop
# ============================================================================
print("\n[Test 9] Mini Training Loop (1 epoch)...")

from torch.optim import Adam

# Create small dataset
train_graphs = generate_synthetic_dataset(n_graphs=20)
train_loader = torch.utils.data.DataLoader(
    train_graphs, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=collator
)

optimizer = Adam(model.parameters(), lr=1e-3)

model.train()
total_loss = 0
for batch_idx, batch in enumerate(train_loader):
    graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
    pyg_batch = batch['pyg_batch'].to(device)
    edge_gt = batch['edge_gt_att'].to(device).float()
    
    optimizer.zero_grad()
    edge_scores, logits = model(graphormer_data, pyg_batch)
    
    exp_loss = F.binary_cross_entropy(edge_scores, edge_gt)
    pred_loss = F.cross_entropy(logits, pyg_batch.y)
    loss = exp_loss + pred_loss
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

avg_loss = total_loss / len(train_loader)
print(f"  Average loss: {avg_loss:.4f}")
print("  ✓ Mini training loop passed")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nThe Graphormer-PIGNN integration is working correctly.")
print("\nNext steps:")
print("1. Run full pre-training: python main/train.py --dataset synthetic")
print("2. Fine-tune on downstream task: python main/train.py --dataset ba2motif --finetune")
print("3. Compare with original π-GNN results")
