# Utility functions for Graphormer-PIGNN
# Includes evaluation metrics, visualization, and helper functions

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple, Optional


def compute_explanation_roc_auc(pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Compute ROC-AUC for explanation task.
    """
    pred_np = pred_scores.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()
    
    try:
        return roc_auc_score(gt_np, pred_np)
    except ValueError:
        return 0.5


def compute_directed_precision(pred_scores: torch.Tensor, gt_labels: torch.Tensor, k: int = None) -> float:
    """
    Compute directed precision: precision of top-k predicted edges.
    """
    pred_np = pred_scores.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()
    
    if k is None:
        k = int(gt_np.sum())
    
    if k == 0:
        return 1.0
    
    top_k_indices = np.argsort(pred_np)[-k:]
    precision = gt_np[top_k_indices].sum() / k
    
    return precision


def compute_explanation_metrics(pred_scores: torch.Tensor, gt_labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute all explanation metrics.
    """
    pred_np = pred_scores.detach().cpu().numpy()
    gt_np = gt_labels.detach().cpu().numpy()
    
    pred_binary = (pred_np > 0.5).astype(int)
    
    try:
        roc_auc = roc_auc_score(gt_np, pred_np)
    except ValueError:
        roc_auc = 0.5
    
    accuracy = (pred_binary == gt_np).mean()
    
    tp = ((pred_binary == 1) & (gt_np == 1)).sum()
    fp = ((pred_binary == 1) & (gt_np == 0)).sum()
    fn = ((pred_binary == 0) & (gt_np == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    dir_precision = compute_directed_precision(pred_scores, gt_labels)
    
    try:
        avg_precision = average_precision_score(gt_np, pred_np)
    except ValueError:
        avg_precision = 0.5
    
    return {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dir_precision': dir_precision,
        'avg_precision': avg_precision
    }


def compute_prediction_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute prediction metrics.
    """
    preds = logits.argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    
    num_classes = logits.size(-1)
    per_class_acc = {}
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            per_class_acc[f'class_{c}_acc'] = (preds[mask] == c).float().mean().item()
    
    return {
        'accuracy': accuracy,
        **per_class_acc
    }


class MetricTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics: Dict[str, float], prefix: str = ''):
        for key, value in metrics.items():
            full_key = f'{prefix}{key}' if prefix else key
            if full_key not in self.metrics:
                self.metrics[full_key] = []
            self.metrics[full_key].append(value)
    
    def get(self, key: str) -> List[float]:
        return self.metrics.get(key, [])
    
    def get_last(self, key: str) -> float:
        values = self.metrics.get(key, [0])
        return values[-1]
    
    def get_best(self, key: str, mode: str = 'max') -> Tuple[float, int]:
        values = self.metrics.get(key, [0])
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        return values[best_idx], best_idx + 1
    
    def save(self, path: str):
        torch.save(self.metrics, path)
    
    def load(self, path: str):
        self.metrics = torch.load(path)


if __name__ == "__main__":
    print("Testing evaluation utilities...")
    
    pred_scores = torch.rand(100)
    gt_labels = (torch.rand(100) > 0.7).long()
    
    metrics = compute_explanation_metrics(pred_scores, gt_labels)
    print(f"Explanation metrics: {metrics}")
    
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    
    pred_metrics = compute_prediction_metrics(logits, targets)
    print(f"Prediction metrics: {pred_metrics}")
    
    print("\nEvaluation utilities test passed!")
