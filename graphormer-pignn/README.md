# Graphormer-PIGNN

Integration of **Graphormer** as the explainer in the **π-GNN** (Pre-training Interpretable GNN) framework.

## Overview

This project combines:
1. **Graphormer** (Transformer for graphs) - for learning rich node representations with structural encodings
2. **π-GNN framework** - for pre-training interpretable GNNs on synthetic graphs with ground-truth explanations

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GraphormerPIGNN                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  GraphormerExplainer                     │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │            GraphormerEncoder                      │   │    │
│  │  │  • Centrality Encoding (degree)                   │   │    │
│  │  │  • Spatial Encoding (shortest path distance)      │   │    │
│  │  │  • Edge Encoding (edge features along paths)      │   │    │
│  │  │  • Multi-head Self-Attention                      │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                        ↓                                 │    │
│  │              Node Representations [N, D]                 │    │
│  │                        ↓                                 │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │           EdgeScorePredictor (MLP)                │   │    │
│  │  │  For each edge (i,j): concat(node_i, node_j)→σ   │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                        ↓                                 │    │
│  │              Edge Scores ρ̂ ∈ [0,1]^|E|                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    GNNPredictor                          │    │
│  │  • GCN layers with edge scores as edge weights           │    │
│  │  • Global mean pooling                                   │    │
│  │  • Classification head                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│                    Class Logits [B, C]                           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Option B for Explainer**: Graphormer learns node representations, then an MLP computes edge scores from node pairs. This only scores **actual edges** (not all node pairs).

2. **Option 2 for Predictor**: A sparse GNN (GCN) uses edge scores as edge weights, naturally respecting graph structure.

3. **No spurious edge weights**: Edge scores are computed only for edges in `edge_index`, ensuring we never assign importance to non-existent edges.

## Installation

```bash
# Clone this repository
git clone <repo_url>
cd graphormer-pignn

# Install dependencies
pip install torch torch_geometric numpy scikit-learn matplotlib

# Optional: for faster shortest path computation
pip install cython
```

## Quick Start

### 1. Test the model

```python
python -c "
from model import GraphormerPIGNN, count_parameters

# Create model
model = GraphormerPIGNN(
    num_encoder_layers=4,
    embedding_dim=128,
    num_attention_heads=4,
    num_classes=3
)

print(f'Model parameters: {count_parameters(model):,}')
"
```

### 2. Pre-train on synthetic data

```bash
python main/train.py \
    --dataset synthetic \
    --n_train 5000 \
    --n_val 1000 \
    --n_test 2000 \
    --epochs 50 \
    --batch_size 32 \
    --num_layers 4 \
    --embedding_dim 128 \
    --exp_weight 1.0 \
    --pred_weight 1.0
```

### 3. Fine-tune on BA-2Motifs (if you have the data)

```bash
python main/train.py \
    --dataset ba2motif \
    --data_dir data/BA2Motif \
    --pretrained_path checkpoints/synthetic_*/best_model.pt \
    --finetune \
    --epochs 30 \
    --batch_size 64 \
    --entropy_reg 0.01
```

## File Structure

```
graphormer-pignn/
├── model.py              # Main model: GraphormerPIGNN, Explainer, Predictor
├── data_utils.py         # Data preprocessing for Graphormer format
├── __init__.py
├── dataset/
│   ├── __init__.py
│   └── datasets.py       # Dataset classes (BA2Motif, Synthetic, etc.)
├── utils/
│   ├── __init__.py
│   └── evaluation.py     # Metrics and evaluation utilities
└── main/
    ├── __init__.py
    └── train.py          # Training script
```

## Training Pipeline

Following the π-GNN paper:

### Phase 1: Pre-training (Explainer)
- Train on synthetic graphs (PT-Motifs) with ground-truth explanations
- Loss: BCE(edge_scores, ground_truth_edge_labels) + CE(predictions, labels)
- The explainer learns universal structural patterns

### Phase 2: Fine-tuning (Explainer + Predictor)
- Load pre-trained explainer weights
- Fine-tune both explainer and predictor on downstream task
- Add entropy regularization for sparser explanations

## Comparing with Original π-GNN

| Component | π-GNN | Graphormer-PIGNN |
|-----------|-------|------------------|
| Node Encoder | SVD + MLP | Graphormer (Transformer) |
| Structural Patterns | Parallel SVD embeddings | Centrality + Spatial + Edge encodings |
| Edge Interaction | Hypergraph convolution | Self-attention (global) |
| Edge Scoring | HyperConv → sigmoid | MLP on node pairs → sigmoid |
| Predictor | GCN with edge weights | GCN with edge weights |

### Expected Differences

**Graphormer-PIGNN advantages:**
- Richer structural encodings (shortest path, degree centrality)
- Global receptive field (attention over all nodes)
- More expressive node representations

**Potential challenges:**
- Higher computational cost (quadratic attention)
- May require more data for pre-training
- More hyperparameters to tune

## Hyperparameters

### Model
- `num_layers`: Number of Graphormer encoder layers (default: 4)
- `embedding_dim`: Hidden dimension (default: 128)
- `num_heads`: Attention heads (default: 4)
- `edge_hidden_dim`: Edge MLP hidden dimension (default: 64)

### Training
- `lr`: Learning rate (default: 1e-4)
- `batch_size`: Batch size (default: 32)
- `exp_weight`: Weight for explanation loss (default: 1.0)
- `pred_weight`: Weight for prediction loss (default: 1.0)
- `entropy_reg`: Entropy regularization (default: 0.0, use > 0 for fine-tuning)

## Citation

If you use this code, please cite:

```bibtex
@article{yin2023pignn,
  title={Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks},
  author={Yin, Jun and Li, Chaozhuo and Yan, Hao and Lian, Jianxun and Wang, Senzhang},
  journal={NeurIPS},
  year={2023}
}

@article{ying2021graphormer,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={NeurIPS},
  year={2021}
}
```

## License

MIT License
