#!/usr/bin/env python
# Training script for Graphormer-PIGNN
# Supports both pre-training on synthetic data and fine-tuning on downstream tasks

import os
import sys
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GraphormerPIGNN, count_parameters
from data_utils import GraphormerCollator, preprocess_pyg_item
from dataset.datasets import BA2Motif, GeneratedSynthetic, generate_synthetic_dataset


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_explanation_metrics(pred_scores, gt_labels):
    """
    Compute explanation metrics.
    
    Args:
        pred_scores: predicted edge importance scores [E]
        gt_labels: ground-truth edge labels [E]
    Returns:
        dict with ROC-AUC, accuracy, precision, recall
    """
    pred_np = pred_scores.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(gt_np, pred_np)
    except ValueError:
        roc_auc = 0.5  # When only one class present
    
    # Binary predictions at threshold 0.5
    pred_binary = (pred_np > 0.5).astype(int)
    accuracy = accuracy_score(gt_np, pred_binary)
    
    # Precision and recall
    tp = ((pred_binary == 1) & (gt_np == 1)).sum()
    fp = ((pred_binary == 1) & (gt_np == 0)).sum()
    fn = ((pred_binary == 0) & (gt_np == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def train_epoch(model, train_loader, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    
    total_exp_loss = 0
    total_pred_loss = 0
    total_loss = 0
    all_exp_scores = []
    all_exp_labels = []
    all_pred_labels = []
    all_pred_targets = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
        pyg_batch = batch['pyg_batch'].to(device)
        edge_gt_att = batch['edge_gt_att']
        
        if edge_gt_att is not None:
            edge_gt_att = edge_gt_att.to(device).float()
        
        optimizer.zero_grad()
        
        # Forward pass
        edge_scores, logits = model(graphormer_data, pyg_batch)
        
        # Explanation loss (BCE)
        if edge_gt_att is not None:
            exp_loss = F.binary_cross_entropy(edge_scores, edge_gt_att)
            all_exp_scores.append(edge_scores.detach())
            all_exp_labels.append(edge_gt_att.detach())
        else:
            exp_loss = torch.tensor(0.0, device=device)
        
        # Prediction loss (CE)
        pred_loss = F.cross_entropy(logits, pyg_batch.y)
        all_pred_labels.append(logits.argmax(dim=1).detach())
        all_pred_targets.append(pyg_batch.y.detach())
        
        # Total loss
        loss = args.exp_weight * exp_loss + args.pred_weight * pred_loss
        
        # Entropy regularization for sparser explanations (during fine-tuning)
        if args.entropy_reg > 0:
            entropy = -edge_scores * torch.log(edge_scores + 1e-8) - \
                      (1 - edge_scores) * torch.log(1 - edge_scores + 1e-8)
            entropy_loss = entropy.mean()
            loss = loss + args.entropy_reg * entropy_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        total_exp_loss += exp_loss.item()
        total_pred_loss += pred_loss.item()
        total_loss += loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}: '
                  f'Loss={loss.item():.4f}, Exp={exp_loss.item():.4f}, Pred={pred_loss.item():.4f}')
    
    # Compute metrics
    n_batches = len(train_loader)
    metrics = {
        'exp_loss': total_exp_loss / n_batches,
        'pred_loss': total_pred_loss / n_batches,
        'total_loss': total_loss / n_batches,
    }
    
    # Explanation metrics
    if all_exp_scores:
        all_exp_scores = torch.cat(all_exp_scores)
        all_exp_labels = torch.cat(all_exp_labels)
        exp_metrics = compute_explanation_metrics(all_exp_scores, all_exp_labels)
        metrics.update({f'exp_{k}': v for k, v in exp_metrics.items()})
    
    # Prediction accuracy
    all_pred_labels = torch.cat(all_pred_labels)
    all_pred_targets = torch.cat(all_pred_targets)
    metrics['pred_accuracy'] = (all_pred_labels == all_pred_targets).float().mean().item()
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, args):
    """Evaluate model."""
    model.eval()
    
    total_exp_loss = 0
    total_pred_loss = 0
    all_exp_scores = []
    all_exp_labels = []
    all_pred_labels = []
    all_pred_targets = []
    
    for batch in loader:
        graphormer_data = {k: v.to(device) for k, v in batch['graphormer_data'].items()}
        pyg_batch = batch['pyg_batch'].to(device)
        edge_gt_att = batch['edge_gt_att']
        
        if edge_gt_att is not None:
            edge_gt_att = edge_gt_att.to(device).float()
        
        edge_scores, logits = model(graphormer_data, pyg_batch)
        
        if edge_gt_att is not None:
            exp_loss = F.binary_cross_entropy(edge_scores, edge_gt_att)
            total_exp_loss += exp_loss.item()
            all_exp_scores.append(edge_scores)
            all_exp_labels.append(edge_gt_att)
        
        pred_loss = F.cross_entropy(logits, pyg_batch.y)
        total_pred_loss += pred_loss.item()
        
        all_pred_labels.append(logits.argmax(dim=1))
        all_pred_targets.append(pyg_batch.y)
    
    n_batches = len(loader)
    metrics = {
        'exp_loss': total_exp_loss / n_batches if all_exp_scores else 0,
        'pred_loss': total_pred_loss / n_batches,
    }
    
    if all_exp_scores:
        all_exp_scores = torch.cat(all_exp_scores)
        all_exp_labels = torch.cat(all_exp_labels)
        exp_metrics = compute_explanation_metrics(all_exp_scores, all_exp_labels)
        metrics.update({f'exp_{k}': v for k, v in exp_metrics.items()})
    
    all_pred_labels = torch.cat(all_pred_labels)
    all_pred_targets = torch.cat(all_pred_targets)
    metrics['pred_accuracy'] = (all_pred_labels == all_pred_targets).float().mean().item()
    
    return metrics


def create_dataloaders(args):
    """Create train/val/test dataloaders."""
    from torch.utils.data import DataLoader
    
    collator = GraphormerCollator(
        max_node=args.max_nodes,
        multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max
    )
    
    if args.dataset == 'synthetic':
        # Generate synthetic data for pre-training
        print("Generating synthetic dataset...")
        train_graphs = generate_synthetic_dataset(
            n_graphs=args.n_train, 
            motif_types=['house', 'cycle', 'star'],
            base_nodes=args.base_nodes
        )
        val_graphs = generate_synthetic_dataset(
            n_graphs=args.n_val,
            motif_types=['house', 'cycle', 'star'],
            base_nodes=args.base_nodes
        )
        test_graphs = generate_synthetic_dataset(
            n_graphs=args.n_test,
            motif_types=['house', 'cycle', 'star'],
            base_nodes=args.base_nodes
        )
        
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, 
                                  shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collator)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)
        
        num_classes = 3
        
    elif args.dataset == 'ba2motif':
        train_dataset = BA2Motif(args.data_dir, mode='train')
        val_dataset = BA2Motif(args.data_dir, mode='valid')
        test_dataset = BA2Motif(args.data_dir, mode='test')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collator)
        
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_loader, val_loader, test_loader, num_classes


def main(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{args.dataset}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(args)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    # Create model
    model = GraphormerPIGNN(
        num_atoms=args.num_atoms,
        num_in_degree=args.num_in_degree,
        num_out_degree=args.num_out_degree,
        num_edges=args.num_edges,
        num_spatial=args.num_spatial,
        num_encoder_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.ffn_dim,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        edge_hidden_dim=args.edge_hidden_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        num_classes=num_classes,
        predictor_layers=args.predictor_layers,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Load pre-trained weights if specified
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pre-trained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        # Load only explainer weights for fine-tuning
        if args.finetune:
            explainer_state = {k.replace('explainer.', ''): v 
                             for k, v in state_dict.items() if k.startswith('explainer.')}
            model.explainer.load_state_dict(explainer_state, strict=False)
        else:
            model.load_state_dict(state_dict)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Training loop
    best_val_metric = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args)
        print(f"\nTrain: Loss={train_metrics['total_loss']:.4f}, "
              f"Exp ROC-AUC={train_metrics.get('exp_roc_auc', 0):.4f}, "
              f"Pred Acc={train_metrics['pred_accuracy']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, args)
        print(f"Val:   Loss={val_metrics['pred_loss']:.4f}, "
              f"Exp ROC-AUC={val_metrics.get('exp_roc_auc', 0):.4f}, "
              f"Pred Acc={val_metrics['pred_accuracy']:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        val_metric = val_metrics.get('exp_roc_auc', 0) + val_metrics['pred_accuracy']
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> New best model saved!")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt'))
    
    # Test with best model
    print(f"\n{'='*60}")
    print(f"Testing with best model from epoch {best_epoch}")
    print('='*60)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device, args)
    
    print(f"\nTest Results:")
    print(f"  Explanation ROC-AUC: {test_metrics.get('exp_roc_auc', 0):.4f}")
    print(f"  Explanation Accuracy: {test_metrics.get('exp_accuracy', 0):.4f}")
    print(f"  Prediction Accuracy: {test_metrics['pred_accuracy']:.4f}")
    
    # Save final results
    results = {
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    torch.save(results, os.path.join(save_dir, 'results.pt'))
    
    print(f"\nResults saved to {save_dir}")
    
    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Graphormer-PIGNN')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ba2motif', 'mutag', 'spmotif'])
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of training graphs (synthetic)')
    parser.add_argument('--n_val', type=int, default=1000, help='Number of validation graphs (synthetic)')
    parser.add_argument('--n_test', type=int, default=2000, help='Number of test graphs (synthetic)')
    parser.add_argument('--base_nodes', type=int, default=20, help='Base graph nodes (synthetic)')
    
    # Model architecture
    parser.add_argument('--num_atoms', type=int, default=512)
    parser.add_argument('--num_in_degree', type=int, default=64)
    parser.add_argument('--num_out_degree', type=int, default=64)
    parser.add_argument('--num_edges', type=int, default=512)
    parser.add_argument('--num_spatial', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Graphormer layers')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--edge_hidden_dim', type=int, default=64)
    parser.add_argument('--predictor_hidden_dim', type=int, default=64)
    parser.add_argument('--predictor_layers', type=int, default=2)
    
    # Data processing
    parser.add_argument('--max_nodes', type=int, default=128)
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--spatial_pos_max', type=int, default=20)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--exp_weight', type=float, default=1.0, help='Weight for explanation loss')
    parser.add_argument('--pred_weight', type=float, default=1.0, help='Weight for prediction loss')
    parser.add_argument('--entropy_reg', type=float, default=0.0, help='Entropy regularization')
    
    # Fine-tuning
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--finetune', action='store_true', help='Fine-tune from pretrained')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=20)
    
    args = parser.parse_args()
    
    main(args)
