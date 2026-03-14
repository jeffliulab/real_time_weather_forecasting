"""
Part 2: Gradient-based saliency analysis for weather forecasting CNN.

Computes saliency maps by backpropagating the output gradient to the input,
then visualizes which geographic regions most influence each prediction target.

Usage:
    python saliency.py --checkpoint runs/cnn_baseline/checkpoints/best.pt
    python saliency.py --checkpoint runs/cnn_baseline/checkpoints/best.pt --n_samples 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data_preparation.dataset import WeatherDataset


TARGET_VARS = [
    "TMP@2m_above_ground",
    "RH@2m_above_ground",
    "UGRD@10m_above_ground",
    "VGRD@10m_above_ground",
    "GUST@surface",
    "APCP_1hr_acc_fcst@surface",
]

TARGET_DISPLAY = ["Temperature", "Humidity", "U-Wind", "V-Wind", "Gust", "Precip."]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute saliency maps")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str,
                        default="/cluster/tufts/c26sp1cs0137/pliu07/assignment2")
    parser.add_argument("--test_year", type=int, default=2020,
                        help="Year to compute saliency on (use val set, not test)")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to average saliency over")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path, device):
    """Load model from a training checkpoint."""
    from models import create_model, MODEL_REGISTRY

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model_name = args["model"]

    metadata = torch.load(Path(args["data_root"]) / "dataset" / "metadata.pt",
                          weights_only=False)
    n_input_channels = metadata["n_vars"]

    from models import get_model_defaults
    defaults = get_model_defaults(model_name)
    n_frames = args.get("n_frames") or defaults["n_frames"]

    model_kwargs = {"n_input_channels": n_input_channels, "n_targets": 6,
                    "base_channels": args.get("base_channels", 64)}
    if n_frames > 1:
        model_kwargs["n_frames"] = n_frames

    model = create_model(model_name, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    norm_stats = ckpt.get("norm_stats")
    return model, norm_stats, args


def compute_saliency_maps(model, dataset, norm_stats, device, n_samples=200, seed=42):
    """
    Compute per-target saliency maps averaged over n_samples.

    Returns:
        saliency: dict mapping target_var -> (H, W) numpy array
        overall: (H, W) numpy array — L2 norm across all targets
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    # Accumulators: per-target saliency
    saliency_sum = {var: None for var in TARGET_VARS}
    overall_sum = None
    count = 0

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        if sample is None:
            continue

        x, target, binary = sample
        x = x.unsqueeze(0).to(device).requires_grad_(True)  # (1, C, H, W)

        pred = model(x)  # (1, 6)

        for j, var in enumerate(TARGET_VARS):
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            pred[0, j].backward(retain_graph=(j < len(TARGET_VARS) - 1))
            grad = x.grad.detach().cpu().squeeze(0)  # (C, H, W)

            spatial_saliency = grad.abs().mean(dim=0).numpy()  # (H, W)

            if saliency_sum[var] is None:
                saliency_sum[var] = np.zeros_like(spatial_saliency)
            saliency_sum[var] += spatial_saliency

        grad_all = x.grad.detach().cpu().squeeze(0)
        overall = torch.norm(grad_all, dim=0).numpy()
        if overall_sum is None:
            overall_sum = np.zeros_like(overall)
        overall_sum += overall

        count += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(indices)} samples")

    saliency = {var: saliency_sum[var] / count for var in TARGET_VARS}
    overall_avg = overall_sum / count

    return saliency, overall_avg


def plot_saliency_maps(saliency, overall, metadata, output_dir):
    """Plot saliency maps overlaid on geographic coordinates."""
    try:
        from cartopy import crs as ccrs
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    grid_x = metadata.get("grid_x")
    grid_y = metadata.get("grid_y")
    jumbo_x_idx = metadata.get("jumbo_x_idx")
    jumbo_y_idx = metadata.get("jumbo_y_idx")

    # Plot all targets + overall
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    all_maps = [(var, saliency[var], name) for var, name in zip(TARGET_VARS, TARGET_DISPLAY)]
    all_maps.append(("overall", overall, "Overall (L2)"))

    for ax_idx, (_, sal_map, title) in enumerate(all_maps):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]

        if grid_x is not None and grid_y is not None:
            im = ax.pcolormesh(grid_x, grid_y, sal_map, cmap="hot", shading="auto")
            if jumbo_x_idx is not None and jumbo_y_idx is not None:
                ax.plot(grid_x[jumbo_x_idx], grid_y[jumbo_y_idx],
                        "c*", markersize=15, markeredgecolor="white", label="Jumbo Statue")
        else:
            im = ax.imshow(sal_map, cmap="hot", aspect="auto", origin="lower")

        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for ax_idx in range(len(all_maps), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Gradient Saliency Maps — Which Regions Drive the Forecast?",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved saliency map plot to {output_dir / 'saliency_maps.png'}")

    # Individual high-res maps for the report
    for var, sal_map, title in all_maps:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if grid_x is not None and grid_y is not None:
            im = ax.pcolormesh(grid_x, grid_y, sal_map, cmap="hot", shading="auto")
            if jumbo_x_idx is not None:
                ax.plot(grid_x[jumbo_x_idx], grid_y[jumbo_y_idx],
                        "c*", markersize=20, markeredgecolor="white")
        else:
            im = ax.imshow(sal_map, cmap="hot", aspect="auto", origin="lower")
        ax.set_title(f"Saliency: {title}", fontsize=14)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        safe_name = var.replace("@", "_at_").replace("/", "_")
        plt.savefig(output_dir / f"saliency_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, norm_stats, train_args = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Model: {train_args['model']}")

    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.checkpoint).parent.parent / "saliency"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (unnormalized — we apply norm in the model input)
    ds = WeatherDataset(args.data_root, [args.test_year], n_frames=1,
                        normalize=True, norm_stats=norm_stats)
    print(f"Dataset: {len(ds)} samples from year {args.test_year}")

    print(f"Computing saliency maps over {args.n_samples} samples...")
    saliency, overall = compute_saliency_maps(
        model, ds, norm_stats, device, n_samples=args.n_samples
    )

    # Save raw saliency data
    torch.save({"saliency": saliency, "overall": overall},
               output_dir / "saliency_data.pt")

    # Load metadata for plotting
    metadata = torch.load(Path(args.data_root) / "dataset" / "metadata.pt",
                          weights_only=False)

    plot_saliency_maps(saliency, overall, metadata, output_dir)

    print("\nDone! Output directory:", output_dir)


if __name__ == "__main__":
    main()
