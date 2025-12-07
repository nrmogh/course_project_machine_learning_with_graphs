#!/usr/bin/env python
"""
Aggregate results from multiple seed runs of the same experiment.
Groups by checkpoint and computes mean/std of test_metric across seeds.

Usage:
    python aggregate_seeds.py \
        --experiment-name sbm_d55 \
        --seeds 42 43 44 45 46 \
        --results-dir . \
        --output results_sbm_d55_aggregated.csv
"""

import argparse
import pandas as pd
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results by checkpoint")
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Base experiment name (without seed suffix)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="List of seeds used"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file"
    )
    
    args = parser.parse_args()
    
    # Load all seed results
    all_dfs = []
    for seed in args.seeds:
        csv_path = os.path.join(args.results_dir, f"results_{args.experiment_name}_seed{seed}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        
        df = pd.read_csv(csv_path)
        df['seed'] = seed
        all_dfs.append(df)
        print(f"Loaded {csv_path}: {len(df)} rows")
    
    if len(all_dfs) == 0:
        print("No data found!")
        return
    
    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Group by checkpoint and compute statistics
    grouped = combined.groupby('checkpoint').agg({
        'epoch': 'first',
        'num_updates': 'first',
        'test_metric': ['mean', 'std', 'min', 'max', 'count'],
        'val_loss': ['mean', 'std'],
        'best_loss': 'first',
        'training_hours': 'mean',
        'metric_name': 'first',
        'mean_edge_homogeneity': 'mean',  
        'mean_avg_degree': 'mean',         
        'mean_degree_variance': 'mean',    
    })
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Rename for clarity
    grouped = grouped.rename(columns={
        'epoch_first': 'epoch',
        'num_updates_first': 'num_updates',
        'test_metric_mean': 'test_mean',
        'test_metric_std': 'test_std',
        'test_metric_min': 'test_min',
        'test_metric_max': 'test_max',
        'test_metric_count': 'n_seeds',
        'val_loss_mean': 'val_mean',
        'val_loss_std': 'val_std',
        'best_loss_first': 'best_loss',
        'training_hours_mean': 'training_hours',
        'metric_name_first': 'metric_name',
        'mean_edge_homogeneity_first': 'edge_homogeneity',
        'mean_avg_degree_first': 'avg_degree',
        'mean_degree_variance_first': 'degree_variance',
    })
    
    # Sort: numbered checkpoints by epoch, then best/last at the end
    def sort_key(row):
        cp = row['checkpoint']
        if cp == 'checkpoint_best.pt':
            return (1, 999999)
        elif cp == 'checkpoint_last.pt':
            return (1, 999998)
        else:
            return (0, row['epoch'] if pd.notna(row['epoch']) else 0)
    
    grouped['_sort'] = grouped.apply(sort_key, axis=1)
    grouped = grouped.sort_values('_sort').drop('_sort', axis=1).reset_index(drop=True)
    
    # Save
    grouped.to_csv(args.output, index=False)
    print(f"\nSaved aggregated results to {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"AGGREGATED RESULTS: {args.experiment_name}")
    print(f"Seeds: {args.seeds}")
    print("=" * 80)
    print(f"{'Checkpoint':<25} | {'Epoch':>6} | {'Test Mean':>12} | {'Test Std':>10} | {'N':>3}")
    print("-" * 80)
    
    for _, row in grouped.iterrows():
        epoch_str = str(int(row['epoch'])) if pd.notna(row['epoch']) else 'N/A'
        std_str = f"{row['test_std']:.6f}" if pd.notna(row['test_std']) else 'N/A'
        print(f"{row['checkpoint']:<25} | {epoch_str:>6} | {row['test_mean']:>12.6f} | {std_str:>10} | {int(row['n_seeds']):>3}")
    
    # Best checkpoint stats
    best_row = grouped[grouped['checkpoint'] == 'checkpoint_best.pt']
    if len(best_row) > 0:
        best_row = best_row.iloc[0]
        print("-" * 80)
        print(f"Best checkpoint: test_mean={best_row['test_mean']:.6f} ± {best_row['test_std']:.6f}")


if __name__ == "__main__":
    main()