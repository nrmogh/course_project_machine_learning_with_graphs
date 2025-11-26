import torch
from collections import OrderedDict

def print_results(ckpt_path):
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    extra = ckpt.get('extra_state', {})
    
    print("=" * 60)
    print("GRAPHORMER TRAINING RESULTS")
    print("=" * 60)
    
    # Best validation loss
    best_loss = extra.get('best', 'N/A')
    val_loss = extra.get('val_loss', 'N/A')
    print(f"\n📊 Best Validation Loss: {best_loss}")
    print(f"📊 Final Validation Loss: {val_loss}")
    
    # Extract metrics from the nested structure
    metrics = extra.get('metrics', {})
    
    # Training metrics
    if 'train' in metrics:
        train_metrics = metrics['train']
        print(f"\n🔥 TRAINING METRICS:")
        
        loss = extract_metric(train_metrics, 'loss')
        if loss:
            print(f"   Final Training Loss: {loss}")
        
        num_updates = extract_metric(train_metrics, 'num_updates')
        if num_updates:
            print(f"   Total Updates: {num_updates}")
        
        lr = extract_metric(train_metrics, 'lr')
        if lr:
            print(f"   Learning Rate: {lr}")
        
        gnorm = extract_metric(train_metrics, 'gnorm')
        if gnorm:
            print(f"   Gradient Norm: {gnorm:.4f}")
        
        train_wall = extract_metric(train_metrics, 'train_wall')
        if train_wall:
            print(f"   Training Time: {train_wall:.2f} seconds ({train_wall/60:.2f} minutes)")
            
            # Calculate speed
            if num_updates:
                ups = num_updates / train_wall
                print(f"   Speed: {ups:.2f} updates/second")
                
                # Time estimate for full training
                total_updates = 400000
                estimated_time = total_updates / ups
                print(f"   Estimated time for 400k updates: {estimated_time/3600:.2f} hours")
        else:
            # Try to get it from extra_state directly
            prev_time = extra.get('previous_training_time', 0)
            if prev_time > 0:
                print(f"   Training Time: {prev_time:.2f} seconds ({prev_time/60:.2f} minutes)")
                if num_updates:
                    ups = num_updates / prev_time
                    print(f"   Speed: {ups:.2f} updates/second")
                    total_updates = 400000
                    estimated_time = total_updates / ups
                    print(f"   Estimated time for 400k updates: {estimated_time/3600:.2f} hours")
    
    # Validation metrics
    if 'valid' in metrics:
        valid_metrics = metrics['valid']
        print(f"\n✅ VALIDATION METRICS:")
        
        val_loss = extract_metric(valid_metrics, 'loss')
        if val_loss:
            print(f"   Validation Loss: {val_loss}")
        
        bsz = extract_metric(valid_metrics, 'bsz')
        if bsz:
            print(f"   Validation Batch Size: {bsz}")
    
    # Training info
    train_iter = extra.get('train_iterator', {})
    print(f"\n📈 TRAINING INFO:")
    print(f"   Completed Epochs: {train_iter.get('epoch', 'N/A')}")
    
    # Model config
    cfg = ckpt.get('cfg', {})
    if 'model' in cfg and hasattr(cfg['model'], 'arch'):
        model_cfg = cfg['model']
        print(f"\n🏗️  MODEL CONFIGURATION:")
        print(f"   Architecture: {model_cfg.arch}")
        print(f"   Encoder Layers: {model_cfg.encoder_layers}")
        print(f"   Embed Dimension: {model_cfg.encoder_embed_dim}")
        print(f"   FFN Dimension: {model_cfg.encoder_ffn_embed_dim}")
        print(f"   Attention Heads: {model_cfg.encoder_attention_heads}")
        print(f"   Dropout: {model_cfg.dropout}")
        print(f"   Attention Dropout: {model_cfg.attention_dropout}")
        print(f"   Batch Size: {model_cfg.batch_size}")
    
    # Dataset info
    if 'task' in cfg:
        task_cfg = cfg['task']
        print(f"\n📦 DATASET:")
        print(f"   Name: {task_cfg.get('dataset_name', 'N/A')}")
        print(f"   Source: {task_cfg.get('dataset_source', 'N/A')}")
        print(f"   Num Classes: {task_cfg.get('num_classes', 'N/A')}")
    
    print("\n" + "=" * 60)
    
    # Clean up
    del ckpt

def extract_metric(metrics_list, key):
    """Extract metric value from the list of tuples structure"""
    for item in metrics_list:
        if len(item) >= 4 and item[1] == key:
            metric_data = item[3]
            if isinstance(metric_data, dict):
                val = metric_data.get('val')
                # Handle StopwatchMeter which stores in 'sum'
                if val is None and 'sum' in metric_data:
                    val = metric_data.get('sum')
                # Convert tensors to float
                if hasattr(val, 'item'):
                    return val.item()
                return val
    return None

if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'ckpts_test-2/checkpoint_best.pt'
    print_results(ckpt_path)