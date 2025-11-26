import torch

ckpt_path = 'ckpts_test/checkpoint_best.pt'

# Load only specific keys to avoid OOM
ckpt = torch.load(ckpt_path, map_location='cpu')

# Immediately extract only what we need, then delete the checkpoint
keys = list(ckpt.keys())
extra_state = ckpt.get('extra_state', {})
cfg = ckpt.get('cfg', None)

# Delete the full checkpoint to free memory
del ckpt

print("=" * 50)
print("TRAINING RESULTS")
print("=" * 50)
print(f"\nCheckpoint keys: {keys}")
print(f"\nExtra state: {extra_state}")
print(f"\nConfig: {cfg}")