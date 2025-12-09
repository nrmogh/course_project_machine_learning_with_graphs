# Graphormer-PIGNN: Integration of Graphormer as Explainer in PI-GNN Framework
# This model uses Graphormer to learn node representations, then computes edge scores
# for explanation, and uses a GNN predictor that respects edge structure.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Optional, Tuple


# ============================================================================
# Graphormer Components (Simplified, standalone version without fairseq dependency)
# ============================================================================

def init_params(module, n_layers):
    """Initialize parameters for Graphormer modules."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class MultiheadAttention(nn.Module):
    """Multi-headed attention with support for attention bias (spatial/edge encoding)."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_bias: [batch, num_heads, seq_len, seq_len]
            key_padding_mask: [batch, seq_len] - True for positions to mask
        Returns:
            output: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project Q, K, V
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]
        
        # Add attention bias (spatial encoding + edge encoding)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights


class GraphormerEncoderLayer(nn.Module):
    """Single Graphormer encoder layer."""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiheadAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            attn_bias: [batch, num_heads, seq_len, seq_len]
            key_padding_mask: [batch, seq_len]
        Returns:
            x: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Pre-norm architecture
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(x, attn_bias, key_padding_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, attn_weights


class GraphNodeFeature(nn.Module):
    """Compute node features including centrality encoding."""
    
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super().__init__()
        
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        
        # [VNode] token for graph-level representation
        self.graph_token = nn.Embedding(1, hidden_dim)
        
        self.apply(lambda m: init_params(m, n_layers))
    
    def forward(self, x, in_degree, out_degree):
        """
        Args:
            x: [batch, n_node, n_features] - node features (as indices)
            in_degree: [batch, n_node]
            out_degree: [batch, n_node]
        Returns:
            node_features: [batch, n_node+1, hidden_dim] (includes graph token)
        """
        batch_size, n_node = x.size()[:2]
        
        # Clamp indices to valid range for embeddings
        x = x.clamp(0, self.atom_encoder.num_embeddings - 1)
        in_degree = in_degree.clamp(0, self.in_degree_encoder.num_embeddings - 1)
        out_degree = out_degree.clamp(0, self.out_degree_encoder.num_embeddings - 1)
        
        # Encode node features
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [B, N, D]
        
        # Add centrality encoding
        node_feature = (
            node_feature 
            + self.in_degree_encoder(in_degree) 
            + self.out_degree_encoder(out_degree)
        )
        
        # Add graph token
        graph_token = self.graph_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        node_feature = torch.cat([graph_token, node_feature], dim=1)
        
        return node_feature


class GraphAttnBias(nn.Module):
    """Compute attention bias from spatial and edge encodings."""
    
    def __init__(self, num_heads, num_edges, num_spatial, hidden_dim, n_layers):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Spatial encoding (based on shortest path distance)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        
        # Edge encoding
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        
        # Virtual node distance
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        
        self.apply(lambda m: init_params(m, n_layers))
    
    def forward(self, attn_bias, spatial_pos, attn_edge_type):
        """
        Args:
            attn_bias: [batch, n_node+1, n_node+1] - base attention bias
            spatial_pos: [batch, n_node, n_node] - shortest path distances
            attn_edge_type: [batch, n_node, n_node, n_edge_features]
        Returns:
            graph_attn_bias: [batch, num_heads, n_node+1, n_node+1]
        """
        batch_size, n_node_plus_1, _ = attn_bias.size()
        n_node = n_node_plus_1 - 1
        
        # Clamp indices to valid range
        spatial_pos = spatial_pos.clamp(0, self.spatial_pos_encoder.num_embeddings - 1)
        attn_edge_type = attn_edge_type.clamp(0, self.edge_encoder.num_embeddings - 1)
        
        # Expand attention bias for all heads
        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Add spatial encoding
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        
        # Add virtual node distance
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        # Add edge encoding
        edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        
        return graph_attn_bias


class GraphormerEncoder(nn.Module):
    """Full Graphormer encoder."""
    
    def __init__(
        self,
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        
        # Node feature and attention bias modules
        self.graph_node_feature = GraphNodeFeature(
            num_atoms, num_in_degree, num_out_degree, 
            embedding_dim, num_encoder_layers
        )
        
        self.graph_attn_bias = GraphAttnBias(
            num_attention_heads, num_edges, num_spatial,
            embedding_dim, num_encoder_layers
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                embedding_dim, num_attention_heads, ffn_embedding_dim,
                dropout, attention_dropout
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batched_data, return_attention=False):
        """
        Args:
            batched_data: dict with keys:
                - x: [batch, n_node, n_features]
                - attn_bias: [batch, n_node+1, n_node+1]
                - spatial_pos: [batch, n_node, n_node]
                - attn_edge_type: [batch, n_node, n_node, n_edge_features]
                - in_degree: [batch, n_node]
                - out_degree: [batch, n_node]
        Returns:
            node_rep: [batch, n_node, embedding_dim] (excluding graph token)
            graph_rep: [batch, embedding_dim]
            all_attn_weights: list of [batch, num_heads, n_node+1, n_node+1] if return_attention
        """
        x = batched_data['x']
        attn_bias = batched_data['attn_bias']
        spatial_pos = batched_data['spatial_pos']
        attn_edge_type = batched_data['attn_edge_type']
        in_degree = batched_data['in_degree']
        out_degree = batched_data['out_degree']
        
        # Compute node features (including graph token)
        x = self.graph_node_feature(x, in_degree, out_degree)  # [B, N+1, D]
        
        # Compute attention bias
        attn_bias = self.graph_attn_bias(attn_bias, spatial_pos, attn_edge_type)
        
        # Compute padding mask
        n_graph, n_node_plus_1 = x.size()[:2]
        padding_mask = (batched_data['x'][:, :, 0] == 0)  # [B, N]
        padding_mask_with_token = torch.cat([
            torch.zeros(n_graph, 1, device=padding_mask.device, dtype=torch.bool),
            padding_mask
        ], dim=1)  # [B, N+1]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_bias, padding_mask_with_token)
            if return_attention:
                all_attn_weights.append(attn_weights)
        
        x = self.layer_norm(x)
        
        # Separate graph token and node representations
        graph_rep = x[:, 0, :]  # [B, D]
        node_rep = x[:, 1:, :]  # [B, N, D]
        
        if return_attention:
            return node_rep, graph_rep, all_attn_weights
        return node_rep, graph_rep


# ============================================================================
# Edge Scoring Module
# ============================================================================

class EdgeScorePredictor(nn.Module):
    """Predict edge importance scores from node representations."""
    
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_rep, edge_index, batch=None):
        """
        Args:
            node_rep: [total_nodes, node_dim] - node representations (unbatched/flattened)
            edge_index: [2, num_edges] - edge indices
            batch: [total_nodes] - batch assignment for each node (optional)
        Returns:
            edge_scores: [num_edges] - importance score for each edge
        """
        # Get source and target node representations
        src_rep = node_rep[edge_index[0]]  # [E, D]
        dst_rep = node_rep[edge_index[1]]  # [E, D]
        
        # Concatenate and predict score
        edge_feat = torch.cat([src_rep, dst_rep], dim=-1)  # [E, 2D]
        edge_scores = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze(-1)  # [E]
        
        return edge_scores


# ============================================================================
# Graphormer Explainer (for PI-GNN framework)
# ============================================================================

class GraphormerExplainer(nn.Module):
    """
    Graphormer-based explainer that outputs edge importance scores.
    
    This replaces the SVDExplainer in the original PI-GNN framework.
    """
    
    def __init__(
        self,
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        edge_hidden_dim=128,
    ):
        super().__init__()
        
        self.graphormer = GraphormerEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        
        self.edge_scorer = EdgeScorePredictor(embedding_dim, edge_hidden_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, batched_data, pyg_batch):
        """
        Args:
            batched_data: dict formatted for Graphormer
            pyg_batch: PyG batch object with edge_index and batch info
        Returns:
            edge_scores: [total_edges] - importance score per edge
            node_rep: [total_nodes, embedding_dim] - node representations
            graph_rep: [batch_size, embedding_dim] - graph representations
        """
        # Get node representations from Graphormer
        node_rep_batched, graph_rep = self.graphormer(batched_data)
        
        # Convert batched representation to flat representation
        # node_rep_batched: [B, max_nodes, D]
        # We need to extract actual nodes (non-padded) for each graph
        
        batch_size, max_nodes, dim = node_rep_batched.size()
        
        # Flatten node representations, keeping only non-padded nodes
        node_rep_list = []
        node_counts = []
        
        x = batched_data['x']  # [B, max_nodes, features]
        for b in range(batch_size):
            # Find non-padded nodes (where x is not all zeros)
            mask = (x[b, :, 0] != 0)  # [max_nodes]
            n_nodes = mask.sum().item()
            node_counts.append(n_nodes)
            node_rep_list.append(node_rep_batched[b, :n_nodes, :])
        
        node_rep = torch.cat(node_rep_list, dim=0)  # [total_nodes, D]
        
        # Compute edge scores for actual edges only
        edge_scores = self.edge_scorer(node_rep, pyg_batch.edge_index)
        
        return edge_scores, node_rep, graph_rep


# ============================================================================
# GNN Predictor (uses edge scores as weights)
# ============================================================================

class GNNPredictor(nn.Module):
    """
    GNN-based predictor that uses edge scores as edge weights.
    
    This respects the graph structure by operating only on actual edges.
    """
    
    def __init__(self, node_in_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_weights, batch):
        """
        Args:
            x: [total_nodes, node_in_dim] - node features
            edge_index: [2, num_edges] - edge indices
            edge_weights: [num_edges] - edge importance scores from explainer
            batch: [total_nodes] - batch assignment
        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.input_proj(x)
        x = F.relu(x)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global mean pooling
        graph_rep = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        return logits


# ============================================================================
# Main Integrated Model
# ============================================================================

class GraphormerPIGNN(nn.Module):
    """
    Full Graphormer-PIGNN model combining:
    1. Graphormer-based explainer for edge importance scores
    2. GNN predictor that uses edge scores as weights
    
    This follows the PI-GNN framework: pre-train explainer on synthetic data,
    then fine-tune both explainer and predictor on downstream tasks.
    """
    
    def __init__(
        self,
        # Graphormer explainer params
        num_atoms=512,
        num_in_degree=512,
        num_out_degree=512,
        num_edges=512,
        num_spatial=512,
        num_encoder_layers=6,
        embedding_dim=256,
        ffn_embedding_dim=256,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        edge_hidden_dim=128,
        # Predictor params
        predictor_hidden_dim=64,
        num_classes=2,
        predictor_layers=2,
    ):
        super().__init__()
        
        self.explainer = GraphormerExplainer(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            edge_hidden_dim=edge_hidden_dim,
        )
        
        self.predictor = GNNPredictor(
            node_in_dim=embedding_dim,
            hidden_dim=predictor_hidden_dim,
            num_classes=num_classes,
            num_layers=predictor_layers,
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, batched_data, pyg_batch, return_explanation=True):
        """
        Args:
            batched_data: dict formatted for Graphormer
            pyg_batch: PyG batch object with edge_index, batch, x, y, etc.
        Returns:
            edge_scores: [total_edges] - edge importance scores (explanation)
            logits: [batch_size, num_classes] - class predictions
        """
        # Get edge scores and node representations from explainer
        edge_scores, node_rep, graph_rep = self.explainer(batched_data, pyg_batch)
        
        # Get predictions using edge scores as weights
        logits = self.predictor(
            node_rep, 
            pyg_batch.edge_index, 
            edge_scores, 
            pyg_batch.batch
        )
        
        if return_explanation:
            return edge_scores, logits
        return logits
    
    def explain_only(self, batched_data, pyg_batch):
        """Get only the edge scores (for evaluation)."""
        edge_scores, _, _ = self.explainer(batched_data, pyg_batch)
        return edge_scores


# ============================================================================
# Utility function to count parameters
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Graphormer-PIGNN Model")
    model = GraphormerPIGNN(
        num_encoder_layers=4,
        embedding_dim=128,
        ffn_embedding_dim=128,
        num_attention_heads=4,
        num_classes=3
    )
    print(f"Total parameters: {count_parameters(model):,}")
