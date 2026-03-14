"""Diagnose why TMP/RH/UGRD/VGRD/GUST RMSE shows nan during training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

ROOT = Path(__file__).parent.parent

# 1. Load norm_stats and checkpoint
print("=== 1. NORM STATS ===")
ns = torch.load(ROOT / "norm_stats.pt", weights_only=True)
for k, v in ns.items():
    print(f"  {k}: {v.flatten().tolist()}")

print("\n=== 2. CHECKPOINT ===")
ckpt_path = ROOT / "runs" / "cnn_baseline" / "checkpoints" / "best.pt"
if not ckpt_path.exists():
    ckpt_path = ROOT / "runs" / "cnn_baseline" / "checkpoints" / "latest.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"  Loaded: {ckpt_path.name}, epoch={ckpt['epoch']}")
ckpt_ns = ckpt.get("norm_stats", {})
print(f"  ckpt target_mean: {ckpt_ns.get('target_mean', 'MISSING')}")
print(f"  ckpt target_std:  {ckpt_ns.get('target_std', 'MISSING')}")

# 2. Load targets and check NaN pattern for validation year
print("\n=== 3. TARGET NaN PATTERN (2020 validation) ===")
tgt = torch.load(ROOT / "dataset" / "targets.pt", weights_only=False)
vals = tgt["values"]  # (26280, 6)
times = tgt["time"]
years = times.astype("datetime64[Y]").astype(int) + 1970
val_mask = years == 2020

val_targets = vals[val_mask]
print(f"  Total 2020 samples: {val_mask.sum()}")
for i, name in enumerate(tgt["variable_names"]):
    col = val_targets[:, i]
    n_nan = torch.isnan(col).sum().item()
    print(f"  [{i}] {name}: nan={n_nan}/{len(col)}")

any_nan = torch.isnan(val_targets).any(dim=1)
print(f"  Samples with ANY nan: {any_nan.sum().item()}")
print(f"  Samples fully valid:  {(~any_nan).sum().item()}")

# 3. Load model and run a few validation samples
print("\n=== 4. MODEL FORWARD PASS CHECK ===")
from models import create_model, get_model_defaults
args = ckpt["args"]
defaults = get_model_defaults(args["model"])
n_frames = args.get("n_frames") or defaults["n_frames"]
metadata = torch.load(ROOT / "dataset" / "metadata.pt", weights_only=False)

model = create_model(args["model"], n_input_channels=metadata["n_vars"],
                     n_targets=6, base_channels=args.get("base_channels", 64))
model.load_state_dict(ckpt["model"])
model.eval()

# Try a few input files
import os
input_dir = ROOT / "dataset" / "inputs" / "2020"
if input_dir.exists():
    files = sorted(os.listdir(input_dir))[:5]
    for fname in files:
        x = torch.load(input_dir / fname, weights_only=True).float()
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        # Normalize
        x_norm = (x - ns["input_mean"]) / (ns["input_std"] + 1e-7)

        with torch.no_grad():
            pred = model(x_norm)

        pred_np = pred.squeeze().numpy()
        nan_count = np.isnan(pred_np).sum()
        inf_count = np.isinf(pred_np).sum()
        print(f"  {fname}: pred={pred_np}, nan={nan_count}, inf={inf_count}")
else:
    print(f"  Input dir not found: {input_dir}")

# 4. Simulate compute_metrics exactly
print("\n=== 5. SIMULATED compute_metrics ===")
from data_preparation.dataset import WeatherDataset

val_ds = WeatherDataset(str(ROOT), [2020], n_frames=1, normalize=True, norm_stats=ns)
print(f"  Val dataset size: {len(val_ds)}")

all_preds, all_targets, all_binary = [], [], []
n_check = min(200, len(val_ds))
print(f"  Running {n_check} samples through model...")

nan_pred_count = 0
for i in range(n_check):
    sample = val_ds[i]
    if sample is None:
        continue
    x, target, binary = sample
    x = x.unsqueeze(0)

    with torch.no_grad():
        pred = model(x)

    if torch.isnan(pred).any():
        nan_pred_count += 1
    if torch.isinf(pred).any():
        print(f"  Sample {i}: inf in prediction!")

    all_preds.append(pred.cpu())
    all_targets.append(target.unsqueeze(0).cpu())
    all_binary.append(binary.unsqueeze(0))

print(f"  Samples with NaN predictions: {nan_pred_count}/{n_check}")

preds = torch.cat(all_preds)
targets = torch.cat(all_targets)
binary_labels = torch.cat(all_binary)

print(f"\n  preds  shape={preds.shape}, nan={torch.isnan(preds).sum().item()}, inf={torch.isinf(preds).sum().item()}")
print(f"  targets shape={targets.shape}, nan={torch.isnan(targets).sum().item()}, inf={torch.isinf(targets).sum().item()}")

# Denormalize
t_mean = ns["target_mean"]
t_std = ns["target_std"]
preds_real = preds * t_std + t_mean
targets_real = targets * t_std + t_mean

print(f"\n  After denormalization:")
print(f"  preds_real  nan={torch.isnan(preds_real).sum().item()}, inf={torch.isinf(preds_real).sum().item()}")
print(f"  targets_real nan={torch.isnan(targets_real).sum().item()}, inf={torch.isinf(targets_real).sum().item()}")

# Per-column NaN counts
print(f"\n  Per-column NaN in preds_real:")
target_vars = tgt["variable_names"]
for j, name in enumerate(target_vars):
    col = preds_real[:, j]
    n_nan = torch.isnan(col).sum().item()
    n_inf = torch.isinf(col).sum().item()
    print(f"    [{j}] {name}: nan={n_nan}, inf={n_inf}, "
          f"min={col[torch.isfinite(col)].min().item():.2f} max={col[torch.isfinite(col)].max().item():.2f}"
          if torch.isfinite(col).any() else f"    [{j}] {name}: ALL nan/inf")

# Valid filter
valid = torch.isfinite(preds_real).all(dim=1) & torch.isfinite(targets_real).all(dim=1)
print(f"\n  Valid samples after filter: {valid.sum().item()}/{len(valid)}")

preds_valid = preds_real[valid]
targets_valid = targets_real[valid]

# Compute RMSE per variable
print(f"\n=== 6. RMSE RESULTS ===")
for j, name in enumerate(target_vars):
    if name == "APCP_1hr_acc_fcst@surface":
        rain_mask = targets_valid[:, j] > 2.0
        n_rain = rain_mask.sum().item()
        if n_rain > 0:
            rmse = torch.sqrt(((preds_valid[rain_mask, j] - targets_valid[rain_mask, j]) ** 2).mean()).item()
        else:
            rmse = float("nan")
        print(f"  [{j}] {name}: RMSE={rmse:.4f} (conditional, n_rain={n_rain})")
    else:
        if len(preds_valid) > 0:
            rmse = torch.sqrt(((preds_valid[:, j] - targets_valid[:, j]) ** 2).mean()).item()
        else:
            rmse = float("nan")
        print(f"  [{j}] {name}: RMSE={rmse:.4f}")
