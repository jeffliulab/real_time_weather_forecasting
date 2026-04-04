"""
Unified training script for weather forecasting models.

Usage:
    python train.py --model cnn_baseline --epochs 30 --batch_size 8
    python train.py --model cnn_multi_frame --n_frames 4 --epochs 30
    python train.py --model cnn_3d --n_frames 4 --epochs 30
    python train.py --model vit --epochs 30 --lr 5e-4
    python train.py --model cnn_baseline --resume checkpoints/cnn_baseline_best.pt
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.data_preparation.dataset import get_dataloaders
from models import create_model, get_model_defaults, MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train weather forecasting model")

    parser.add_argument("--model", type=str, default="cnn_baseline",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data_root", type=str,
                        default="/cluster/tufts/c26sp1cs0137/pliu07/assignment2")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"])

    # Model-specific
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Override default n_frames for the model")
    parser.add_argument("--base_channels", type=int, default=64)

    # Infrastructure
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def setup_output_dir(args):
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = Path(args.data_root) / "runs" / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    return out


def get_device(args):
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricLogger:
    """Log training metrics to CSV and generate plots."""

    def __init__(self, log_dir, target_vars):
        self.log_path = log_dir / "training_log.csv"
        self.target_vars = target_vars
        self.history = []

        header = ["epoch", "train_loss", "val_loss"]
        header += [f"val_rmse_{v}" for v in target_vars]
        header += ["val_rmse_apcp_rain", "val_auc", "lr", "epoch_time"]
        self.header = header

        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def log(self, metrics):
        self.history.append(metrics)
        row = [metrics.get(h, "") for h in self.header]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def plot(self, fig_dir):
        if len(self.history) < 2:
            return
        epochs = [m["epoch"] for m in self.history]
        train_loss = [m["train_loss"] for m in self.history]
        val_loss = [m["val_loss"] for m in self.history]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, train_loss, label="Train")
        axes[0].plot(epochs, val_loss, label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for var in self.target_vars:
            key = f"val_rmse_{var}"
            vals = [m.get(key, float("nan")) for m in self.history]
            axes[1].plot(epochs, vals, label=var.split("@")[0])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Validation RMSE by Variable")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=150)
        plt.close()


def compute_metrics(preds, targets, binary_labels, norm_stats, target_vars):
    """Compute per-variable RMSE, conditional APCP RMSE, and AUC.

    NaN filtering is done per-variable: targets.pt has NaN in TMP/RH/UGRD/VGRD/GUST
    (~2% of samples) but zero NaN in APCP. Filtering per-variable avoids discarding
    valid data for one variable because another variable is NaN.
    """
    from sklearn.metrics import roc_auc_score

    t_mean = norm_stats["target_mean"]
    t_std = norm_stats["target_std"]
    preds_real = preds * t_std + t_mean
    targets_real = targets * t_std + t_mean

    metrics = {}
    apcp_idx = target_vars.index("APCP_1hr_acc_fcst@surface")

    for j, var in enumerate(target_vars):
        p = preds_real[:, j]
        t = targets_real[:, j]
        valid_j = torch.isfinite(p) & torch.isfinite(t)
        p_valid = p[valid_j]
        t_valid = t[valid_j]

        if var == "APCP_1hr_acc_fcst@surface":
            rain_mask = t_valid > 2.0
            n_rain = rain_mask.sum().item()
            if n_rain > 0:
                rmse = torch.sqrt(((p_valid[rain_mask] - t_valid[rain_mask]) ** 2).mean()).item()
            else:
                rmse = float("nan")
            metrics[f"val_rmse_{var}"] = rmse
            metrics["val_rmse_apcp_rain"] = rmse
        else:
            if len(p_valid) > 0:
                rmse = torch.sqrt(((p_valid - t_valid) ** 2).mean()).item()
            else:
                rmse = float("nan")
            metrics[f"val_rmse_{var}"] = rmse

    # AUC — use all samples where APCP prediction is finite
    apcp_valid = torch.isfinite(preds_real[:, apcp_idx])
    scores = preds_real[apcp_valid, apcp_idx].numpy()
    labels = binary_labels[apcp_valid].numpy().astype(int)
    if len(labels) > 0 and labels.sum() > 0 and (1 - labels).sum() > 0 and np.isfinite(scores).all():
        metrics["val_auc"] = roc_auc_score(labels, scores)
    else:
        metrics["val_auc"] = float("nan")

    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    total_loss = 0
    n_batches = 0
    n_total = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        x, target, _ = batch
        x, target = x.to(device), target.to(device)

        pred = model(x)
        loss = criterion(pred, target)

        if torch.isnan(loss):
            print(f"  [epoch {epoch}] NaN loss at batch {batch_idx}, skipping")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        del x, target, pred, loss

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_total:
            avg = total_loss / n_batches if n_batches > 0 else float("nan")
            print(f"  [train] batch {batch_idx+1}/{n_total} loss={avg:.4f}", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    # Accumulate only small CPU tensors (predictions are (B,6), not full inputs)
    all_preds, all_targets, all_binary = [], [], []
    n_total = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        x, target, binary = batch
        x, target = x.to(device), target.to(device)

        pred = model(x)
        loss = criterion(pred, target)

        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())
        all_binary.append(binary)

        # Free GPU memory for large multi-frame inputs
        del x, target, pred, loss

        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == n_total:
            print(f"  [val] batch {batch_idx+1}/{n_total}", flush=True)

    val_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    binary = torch.cat(all_binary)

    return val_loss, preds, targets, binary


def main():
    args = parse_args()
    device = get_device(args)
    out_dir = setup_output_dir(args)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # Model defaults
    defaults = get_model_defaults(args.model)
    n_frames = args.n_frames or defaults["n_frames"]
    stack_mode = defaults["stack_mode"]

    # Data
    train_loader, val_loader, norm_stats = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        n_frames=n_frames,
        stack_mode=stack_mode,
        num_workers=args.num_workers,
    )

    metadata = torch.load(Path(args.data_root) / "dataset" / "metadata.pt", weights_only=False)
    target_vars = [
        "TMP@2m_above_ground", "RH@2m_above_ground",
        "UGRD@10m_above_ground", "VGRD@10m_above_ground",
        "GUST@surface", "APCP_1hr_acc_fcst@surface",
    ]

    # Model
    n_input_channels = metadata["n_vars"]
    model_kwargs = {"n_input_channels": n_input_channels, "n_targets": 6,
                    "base_channels": args.base_channels}
    if n_frames > 1:
        model_kwargs["n_frames"] = n_frames

    model = create_model(args.model, **model_kwargs)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    else:
        scheduler = None

    criterion = nn.MSELoss()

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    logger = MetricLogger(out_dir / "logs", target_vars)

    # Move norm_stats to CPU for metric computation
    norm_stats_cpu = {k: v.cpu() if torch.is_tensor(v) else v
                      for k, v in norm_stats.items()}

    epochs_no_improve = 0

    print(f"\nTraining for {args.epochs} epochs...")
    if args.patience > 0:
        print(f"Early stopping enabled: patience={args.patience}")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, preds, targets, binary = validate(model, val_loader, criterion, device)

        metrics = compute_metrics(preds, targets, binary, norm_stats_cpu, target_vars)
        metrics.update({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - t0,
        })

        logger.log(metrics)
        logger.plot(out_dir / "figures")

        # Scheduler step
        if args.scheduler == "cosine" and scheduler:
            scheduler.step()
        elif args.scheduler == "plateau" and scheduler:
            scheduler.step(val_loss)

        # Checkpointing
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "args": vars(args),
            "norm_stats": norm_stats_cpu,
        }
        torch.save(ckpt_data, out_dir / "checkpoints" / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(ckpt_data, out_dir / "checkpoints" / "best.pt")
            marker = " *best*"
        else:
            epochs_no_improve += 1
            marker = ""

        # Print summary
        rmse_strs = []
        for var in target_vars:
            key = f"val_rmse_{var}"
            val = metrics.get(key, float("nan"))
            short = var.split("@")[0][:6]
            rmse_strs.append(f"{short}={val:.3f}")

        auc_str = f"AUC={metrics.get('val_auc', float('nan')):.3f}"

        print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"{' '.join(rmse_strs)} {auc_str} | "
              f"lr={optimizer.param_groups[0]['lr']:.1e} "
              f"t={time.time()-t0:.0f}s{marker}")

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    print(f"Logs: {out_dir / 'logs'}")
    print(f"Figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
