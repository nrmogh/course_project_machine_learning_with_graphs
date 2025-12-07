import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
import csv
import logging

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from pretrain import load_pretrained_model


CSV_PATH = "eval_results.csv"


def ensure_csv_header():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint",
                "epoch",
                "num_updates",
                "val_loss",
                "best_loss",
                "test_metric",
                "metric_name",
                "training_hours",
                "mean_edge_homogeneity",
                "mean_avg_degree",
                "mean_degree_variance",
            ])


def append_csv_row(**kwargs):
    """Append a single row to CSV immediately."""
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            kwargs.get("checkpoint", ""),
            kwargs.get("epoch", ""),
            kwargs.get("num_updates", ""),
            kwargs.get("val_loss", ""),
            kwargs.get("best_loss", ""),
            kwargs.get("test_metric", ""),
            kwargs.get("metric_name", ""),
            kwargs.get("training_hours", ""),
            kwargs.get("mean_edge_homogeneity", ""),
            kwargs.get("mean_avg_degree", ""),
            kwargs.get("mean_degree_variance", ""),
        ])


def print_checkpoint_info(checkpoint_path):
    """Print training progress information from checkpoint and return info."""
    out = {
        "epoch": None,
        "num_updates": None,
        "val_loss": None,
        "best_loss": None,
        "training_hours": None,
    }

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        extra = ckpt.get('extra_state', {})

        train_iter = extra.get('train_iterator', {})
        out["epoch"] = train_iter.get('epoch', None)

        metrics = extra.get('metrics', {})
        if "train" in metrics:
            train_metrics = metrics["train"]
            for item in train_metrics:
                if len(item) >= 4 and item[1] == "num_updates":
                    out["num_updates"] = item[3].get("val")

        out["val_loss"] = extra.get("val_loss", None)
        out["best_loss"] = extra.get("best", None)

        prev_time = extra.get("previous_training_time", 0)
        if prev_time > 0:
            out["training_hours"] = prev_time / 3600

        del ckpt
    except Exception as e:
        print(f"Could not read checkpoint info from {checkpoint_path}: {e}")

    return out


def eval_checkpoint(cfg, task, model, checkpoint_path, use_pretrained, split, metric):
    """Evaluate a single checkpoint using pre-loaded task."""
    
    # Load model weights
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]

    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)
    del model_state

    model.to(torch.cuda.current_device())

    # Get batch iterator (dataset is already loaded in task)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )

    itr = batch_iterator.next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    y_pred, y_true = [], []
    with torch.no_grad():
        model.eval()
        for sample in progress:
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    # Debug output
    print(f"y_true range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # Compute metric
    if metric == "auc":
        metric_value = roc_auc_score(y_true.numpy(), y_pred.numpy())
        metric_name = "auc"
    elif metric == "smse":
        # Standardized MSE
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        var = np.var(y_true_np)
        metric_value = mse / var if var > 0 else mse
        metric_name = "smse"
    else:
        metric_value = np.mean(np.abs(y_true.numpy() - y_pred.numpy()))
        metric_name = "mae"

    # Compute mean edge homogeneity from dataset (if available)
    mean_edge_homogeneity = None
    mean_avg_degree = None
    mean_degree_variance = None
    try:
        dataset = task.dataset(split)
        # Try to access underlying dataset
        if hasattr(dataset, 'dataset'):
            inner_dataset = dataset.dataset
        else:
            inner_dataset = dataset
        
        # Check if it's a DCSBM dataset with graph statistics
        homogeneities = []
        avg_degrees = []
        degree_variances = []
        for i in range(min(len(inner_dataset), 1000)):  # Sample up to 1000
            data = inner_dataset[i]
            if hasattr(data, 'edge_homogeneity'):
                homogeneities.append(data.edge_homogeneity)
            if hasattr(data, 'avg_degree'):
                avg_degrees.append(data.avg_degree)
            if hasattr(data, 'degree_variance'):
                degree_variances.append(data.degree_variance)
        
        if homogeneities:
            mean_edge_homogeneity = np.mean(homogeneities)
            print(f"Mean edge homogeneity: {mean_edge_homogeneity:.4f}")
        if avg_degrees:
            mean_avg_degree = np.mean(avg_degrees)
            print(f"Mean avg degree: {mean_avg_degree:.4f}")
        if degree_variances:
            mean_degree_variance = np.mean(degree_variances)
            print(f"Mean degree variance: {mean_degree_variance:.4f}")
    except Exception as e:
        print(f"Could not compute graph statistics: {e}")

    return metric_value, metric_name, mean_edge_homogeneity, mean_avg_degree, mean_degree_variance


def main():
    parser = options.get_training_parser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="eval_results.csv",
        help="Where to save evaluation results (CSV file)."
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    global CSV_PATH
    CSV_PATH = args.csv_path
  
    logger = logging.getLogger(__name__)

    ensure_csv_header()

    # Convert args to config
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # Setup task and load dataset ONCE
    print("[Evaluate] Setting up task and loading dataset...")
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    
    # Load the split we want to evaluate on
    split = args.split
    task.load_dataset(split)
    print(f"[Evaluate] Dataset loaded. Evaluating on '{split}' split.")

    if args.pretrained_model_name != "none":
        metric_value, metric_name, mean_edge_homogeneity, mean_avg_degree, mean_degree_variance = eval_checkpoint(
            cfg, task, model, None, True, split, args.metric
        )
        append_csv_row(
            checkpoint="pretrained",
            epoch="n/a",
            num_updates="n/a",
            val_loss="n/a",
            best_loss="n/a",
            test_metric=metric_value,
            metric_name=metric_name,
            training_hours="n/a",
            mean_edge_homogeneity=mean_edge_homogeneity,
            mean_avg_degree=mean_avg_degree,
            mean_degree_variance=mean_degree_variance,
        )
    else:
        checkpoint_files = sorted([
            f for f in os.listdir(args.save_dir) 
            if f.endswith('.pt')
        ])
        
        print(f"[Evaluate] Found {len(checkpoint_files)} checkpoints to evaluate.")
        
        # Compute graph stats once (dataset property, not checkpoint-dependent)
        mean_edge_homogeneity = None
        mean_avg_degree = None
        mean_degree_variance = None
        
        for i, checkpoint_fname in enumerate(checkpoint_files):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            print(f"\n[Evaluate] Checkpoint {i+1}/{len(checkpoint_files)}: {checkpoint_fname}")

            # Extract training metadata
            info = print_checkpoint_info(checkpoint_path)

            # Evaluate (reuses task/dataset)
            metric_value, metric_name, edge_hom, avg_deg, deg_var = eval_checkpoint(
                cfg, task, model, checkpoint_path, False, split, args.metric
            )
            
            # Store graph stats from first checkpoint (same for all)
            if mean_edge_homogeneity is None:
                mean_edge_homogeneity = edge_hom
                mean_avg_degree = avg_deg
                mean_degree_variance = deg_var

            # Log to CSV immediately
            append_csv_row(
                checkpoint=checkpoint_fname,
                epoch=info["epoch"],
                num_updates=info["num_updates"],
                val_loss=info["val_loss"],
                best_loss=info["best_loss"],
                test_metric=metric_value,
                metric_name=metric_name,
                training_hours=info["training_hours"],
                mean_edge_homogeneity=mean_edge_homogeneity,
                mean_avg_degree=mean_avg_degree,
                mean_degree_variance=mean_degree_variance,
            )
            
            print(f"[Evaluate] {checkpoint_fname}: {metric_name}={metric_value:.6f}")


if __name__ == "__main__":
    main()