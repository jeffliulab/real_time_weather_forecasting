#!/usr/bin/env python3
"""
Evaluate all Part 1 models on the test set and produce a comparison table.

Usage:
    PYTHONUNBUFFERED=1 python evaluation/evaluate_all.py

Evaluates: stub (persistence), baseline_cnn, resnet18, convnext_tiny
on the 2021 test set.
"""

import sys
import importlib.util
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ============================================================
# Configuration
# ============================================================

MODELS = ["stub", "baseline_cnn", "resnet18", "convnext_tiny", "vit", "cnn_multi_frame", "cnn_3d"]
TEST_YEAR = 2021

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent
DATASET_DIR = ROOT / "dataset"

# ============================================================
# Load dataset
# ============================================================

print("Loading dataset metadata and targets ...", flush=True)
metadata = torch.load(DATASET_DIR / "metadata.pt", weights_only=False)
targets_data = torch.load(DATASET_DIR / "targets.pt", weights_only=False)

times = targets_data["time"]
target_values = targets_data["values"]
binary_labels = targets_data["binary_label"]
target_vars = list(targets_data["variable_names"])

print(f"  Full dataset : {times[0]}  ->  {times[-1]}  ({len(times)} steps)", flush=True)
print(f"  Target vars  : {target_vars}", flush=True)

# ============================================================
# Test indices
# ============================================================

times_years = times.astype("datetime64[Y]").astype(int) + 1970
in_year_mask = np.where(times_years == TEST_YEAR)[0]
test_indices = in_year_mask[in_year_mask + 24 < len(times)]

print(f"\nTest year: {TEST_YEAR}, # samples: {len(test_indices)}", flush=True)


def load_input(t_idx):
    """Load a single input tensor on-the-fly."""
    dt = pd.Timestamp(times[t_idx])
    input_path = (DATASET_DIR / "inputs" / str(dt.year)
                  / f"X_{dt.strftime('%Y%m%d%H')}.pt")
    if not input_path.exists():
        return None
    x = torch.load(input_path, weights_only=True).float()
    if torch.isnan(x).any():
        return None
    return x


# ============================================================
# Evaluate each model
# ============================================================

apcp_var = "APCP_1hr_acc_fcst@surface"
apcp_idx = target_vars.index(apcp_var)

all_results = {}

for model_name in MODELS:
    print(f"\n{'='*65}", flush=True)
    print(f"Evaluating: {model_name}", flush=True)
    print(f"{'='*65}", flush=True)

    model_path = EVAL_DIR / model_name / "model.py"
    if not model_path.exists():
        print(f"  SKIP: {model_path} not found", flush=True)
        continue

    # Load model
    model_dir = str(EVAL_DIR / model_name)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location(f"model_{model_name}", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.get_model(metadata)
    model.eval()
    print(f"  Model class: {model.__class__.__name__}", flush=True)

    # Check if multi-frame model
    n_frames = getattr(module, "N_FRAMES", 1)
    if n_frames > 1:
        print(f"  Multi-frame model: n_frames={n_frames}", flush=True)

    # Inference
    all_preds = []
    all_targets = []
    all_binary = []
    skipped = 0

    with torch.no_grad():
        for step, t_idx in enumerate(test_indices):
            t24_idx = t_idx + 24

            if n_frames == 1:
                x = load_input(t_idx)
                if x is None:
                    skipped += 1
                    continue
                x_batch = x.unsqueeze(0)  # (1, H, W, C)
            else:
                # Load k consecutive frames: t-(k-1), ..., t-1, t
                frames = []
                skip_sample = False
                for offset in range(-(n_frames - 1), 1):
                    frame = load_input(t_idx + offset)
                    if frame is None:
                        skip_sample = True
                        break
                    frames.append(frame)
                if skip_sample:
                    skipped += 1
                    continue
                x_batch = torch.stack(frames, dim=0).unsqueeze(0)  # (1, k, H, W, C)

            pred = model(x_batch).squeeze(0)
            all_preds.append(pred)
            all_targets.append(target_values[t24_idx])
            all_binary.append(binary_labels[t24_idx])

            if (step + 1) % 2000 == 0 or (step + 1) == len(test_indices):
                print(f"  {step+1:>5}/{len(test_indices)} (skipped {skipped})", flush=True)

    preds = torch.stack(all_preds).float()
    targets = torch.stack(all_targets).float()
    binary = torch.stack(all_binary)
    print(f"  Evaluated {len(all_preds)} samples (skipped {skipped})", flush=True)

    # Compute metrics
    results = {}
    for j, var in enumerate(target_vars):
        p = preds[:, j]
        t = targets[:, j]
        valid_mask = torch.isfinite(p) & torch.isfinite(t)
        p_valid = p[valid_mask]
        t_valid = t[valid_mask]

        if var == apcp_var:
            rain_mask = t_valid > 2.0
            n_rain = rain_mask.sum().item()
            if n_rain > 0:
                rmse = torch.sqrt(((p_valid[rain_mask] - t_valid[rain_mask]) ** 2).mean()).item()
            else:
                rmse = float("nan")
            results[var] = rmse
        else:
            if len(p_valid) > 0:
                rmse = torch.sqrt(((p_valid - t_valid) ** 2).mean()).item()
            else:
                rmse = float("nan")
            results[var] = rmse

    # AUC
    scores = preds[:, apcp_idx].numpy()
    labels = binary.numpy().astype(int)
    if labels.sum() > 0 and (1 - labels).sum() > 0 and np.isfinite(scores).all():
        results["AUC"] = roc_auc_score(labels, scores)
    else:
        results["AUC"] = float("nan")

    all_results[model_name] = results

    # Print individual results
    for var in target_vars:
        label = "RMSE (>2mm)" if var == apcp_var else "RMSE"
        print(f"  {var:42s}  {label}: {results[var]:8.4f}", flush=True)
    print(f"  {'AUC':42s}  {results['AUC']:.4f}", flush=True)

# ============================================================
# Comparison table
# ============================================================

print(f"\n\n{'='*90}", flush=True)
print(f"COMPARISON TABLE  --  Test Year: {TEST_YEAR}", flush=True)
print(f"{'='*90}", flush=True)

# Header
short_names = {
    "TMP@2m_above_ground": "TMP",
    "RH@2m_above_ground": "RH",
    "UGRD@10m_above_ground": "UGRD",
    "VGRD@10m_above_ground": "VGRD",
    "GUST@surface": "GUST",
    "APCP_1hr_acc_fcst@surface": "APCP(>2mm)",
}

header = f"{'Model':<20s}"
for var in target_vars:
    header += f" {short_names[var]:>10s}"
header += f" {'AUC':>8s}"
print(header, flush=True)
print("-" * 90, flush=True)

# Find best (lowest) per column for highlighting
best_per_col = {}
for var in target_vars:
    vals = [all_results[m].get(var, float("inf")) for m in all_results]
    best_per_col[var] = min(v for v in vals if not np.isnan(v))
auc_vals = [all_results[m].get("AUC", 0) for m in all_results]
best_per_col["AUC"] = max(v for v in auc_vals if not np.isnan(v))

for model_name in MODELS:
    if model_name not in all_results:
        continue
    res = all_results[model_name]
    row = f"{model_name:<20s}"
    for var in target_vars:
        val = res.get(var, float("nan"))
        marker = " *" if abs(val - best_per_col[var]) < 1e-6 else "  "
        row += f" {val:>8.4f}{marker}"
    auc = res.get("AUC", float("nan"))
    marker = " *" if abs(auc - best_per_col["AUC"]) < 1e-6 else "  "
    row += f" {auc:>6.4f}{marker}"
    print(row, flush=True)

print("-" * 90, flush=True)
print("* = best in column", flush=True)
print(f"{'='*90}", flush=True)
