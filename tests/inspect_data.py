"""Inspect dataset metadata, targets, and norm_stats on the remote."""
import torch
import numpy as np

# 1. metadata
meta = torch.load("dataset/metadata.pt", weights_only=False)
print("=== METADATA ===")
for k, v in meta.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")
        if len(v) <= 50:
            for i, item in enumerate(v):
                print(f"    [{i}] {item}")
    else:
        print(f"  {k}: {v}")

# 2. targets
tgt = torch.load("dataset/targets.pt", weights_only=False)
print("\n=== TARGETS ===")
for k, v in tgt.items():
    if isinstance(v, torch.Tensor):
        n_nan = torch.isnan(v.float()).sum().item()
        print(f"  {k}: shape={v.shape} dtype={v.dtype} nan={n_nan}")
        if v.ndim <= 1 or v.shape[-1] <= 10:
            print(f"    min={v.float().min():.4f} max={v.float().max():.4f}")
    elif isinstance(v, np.ndarray):
        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
    elif isinstance(v, list):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v}")

# per-var stats for targets
vals = tgt["values"]
names = tgt["variable_names"]
print("\n  Per-variable target stats:")
for i, name in enumerate(names):
    col = vals[:, i]
    n_nan = torch.isnan(col).sum().item()
    valid = col[~torch.isnan(col)]
    if len(valid) > 0:
        print(f"    [{i}] {name}: mean={valid.mean():.2f} std={valid.std():.2f} "
              f"min={valid.min():.2f} max={valid.max():.2f} nan={n_nan}/{len(col)}")
    else:
        print(f"    [{i}] {name}: ALL NaN ({n_nan})")

# 3. norm_stats
print("\n=== NORM_STATS ===")
ns = torch.load("norm_stats.pt", weights_only=True)
for k, v in ns.items():
    n_nan = torch.isnan(v).sum().item()
    print(f"  {k}: shape={v.shape} dtype={v.dtype} nan={n_nan}")
    print(f"    min={v.min():.6f} max={v.max():.6f}")
    if "target" in k:
        print(f"    values={v.flatten().tolist()}")

# 4. Check a sample input
import os
sample_dir = "dataset/inputs/2018"
if os.path.isdir(sample_dir):
    files = sorted(os.listdir(sample_dir))[:3]
    print(f"\n=== SAMPLE INPUTS (first 3 from {sample_dir}) ===")
    for fname in files:
        x = torch.load(os.path.join(sample_dir, fname), weights_only=True).float()
        n_nan = torch.isnan(x).sum().item()
        print(f"  {fname}: shape={x.shape} dtype=bfloat16->float32 "
              f"min={x.min():.2f} max={x.max():.2f} nan={n_nan}/{x.numel()}")
