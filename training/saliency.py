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
    # Ensure norm_stats are on CPU for dataset loading
    if norm_stats is not None:
        norm_stats = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in norm_stats.items()}
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

    # Spatial analysis: directional and distance-based saliency breakdown
    analyze_spatial_saliency(saliency, overall, metadata, output_dir)

    print("\nDone! Output directory:", output_dir)


def analyze_spatial_saliency(saliency, overall, metadata, output_dir):
    """
    Quantitative spatial analysis of saliency maps:
    - Directional breakdown (N/S/E/W quadrants relative to Jumbo)
    - Distance-based decay from Jumbo
    - Top contributing regions
    """
    jumbo_y = metadata["jumbo_y_idx"]  # row index
    jumbo_x = metadata["jumbo_x_idx"]  # col index
    grid_x = metadata.get("grid_x")
    grid_y = metadata.get("grid_y")
    H, W = overall.shape

    lines = []
    lines.append("=" * 70)
    lines.append("SPATIAL SALIENCY ANALYSIS")
    lines.append(f"Jumbo Statue grid position: row={jumbo_y}, col={jumbo_x}")
    lines.append("=" * 70)

    # --- Quadrant analysis (relative to Jumbo) ---
    north = overall[:jumbo_y, :]       # rows above Jumbo (lower y-index = south in image coords, but grid_y increasing = north)
    south = overall[jumbo_y:, :]
    west = overall[:, :jumbo_x]
    east = overall[:, jumbo_x:]

    # In Lambert conformal: grid_y increases northward
    # So rows 0..jumbo_y = south, rows jumbo_y..H = north
    # grid_x increases eastward: cols 0..jumbo_x = west, cols jumbo_x..W = east
    if grid_y is not None and len(grid_y) > 1:
        if grid_y[0] < grid_y[-1]:
            # grid_y increasing with row index = row 0 is south
            south_sal = overall[:jumbo_y, :].mean()
            north_sal = overall[jumbo_y:, :].mean()
        else:
            south_sal = overall[jumbo_y:, :].mean()
            north_sal = overall[:jumbo_y, :].mean()
    else:
        south_sal = overall[jumbo_y:, :].mean()
        north_sal = overall[:jumbo_y, :].mean()

    west_sal = overall[:, :jumbo_x].mean()
    east_sal = overall[:, jumbo_x:].mean()
    total_sal = overall.mean()

    lines.append("\n--- Directional Saliency (mean, relative to Jumbo) ---")
    lines.append(f"  North: {north_sal:.6f}  ({north_sal/total_sal:.2f}x avg)")
    lines.append(f"  South: {south_sal:.6f}  ({south_sal/total_sal:.2f}x avg)")
    lines.append(f"  West:  {west_sal:.6f}  ({west_sal/total_sal:.2f}x avg)")
    lines.append(f"  East:  {east_sal:.6f}  ({east_sal/total_sal:.2f}x avg)")
    lines.append(f"  Overall mean: {total_sal:.6f}")

    # Physical interpretation hint
    if west_sal > east_sal:
        lines.append("  -> Western regions contribute more than eastern (consistent with prevailing westerlies)")
    if south_sal > north_sal:
        lines.append("  -> Southern regions contribute more (possible subtropical moisture transport)")

    # --- Distance-based analysis ---
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((yy - jumbo_y) ** 2 + (xx - jumbo_x) ** 2)

    # Distance bins (in grid cells, ~3km each)
    bin_edges = [0, 25, 50, 100, 150, 200, 300]
    lines.append("\n--- Saliency vs Distance from Jumbo ---")
    lines.append(f"  {'Range (cells)':>18} {'~km':>8} {'Mean Sal':>12} {'Rel':>8}")
    for i in range(len(bin_edges) - 1):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_sal = overall[mask].mean()
            km_lo, km_hi = bin_edges[i] * 3, bin_edges[i + 1] * 3
            lines.append(f"  {bin_edges[i]:>6}-{bin_edges[i+1]:<6}      {km_lo:>3}-{km_hi:<3}km  {mean_sal:.6f}  {mean_sal/total_sal:>6.2f}x")

    # --- Per-target directional breakdown ---
    lines.append("\n--- Per-Target Directional Breakdown ---")
    lines.append(f"  {'Target':<20} {'North':>10} {'South':>10} {'West':>10} {'East':>10}")
    for var, name in zip(TARGET_VARS, TARGET_DISPLAY):
        sal = saliency[var]
        if grid_y is not None and len(grid_y) > 1 and grid_y[0] < grid_y[-1]:
            n = sal[jumbo_y:, :].mean()
            s = sal[:jumbo_y, :].mean()
        else:
            n = sal[:jumbo_y, :].mean()
            s = sal[jumbo_y:, :].mean()
        w = sal[:, :jumbo_x].mean()
        e = sal[:, jumbo_x:].mean()
        lines.append(f"  {name:<20} {n:>10.6f} {s:>10.6f} {w:>10.6f} {e:>10.6f}")

    report = "\n".join(lines)
    print(report)

    with open(output_dir / "saliency_analysis.txt", "w") as f:
        f.write(report)
    print(f"\nSaved analysis to {output_dir / 'saliency_analysis.txt'}")

    # --- Distance decay plot ---
    n_bins = 50
    max_dist = dist.max()
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        lo = max_dist * i / n_bins
        hi = max_dist * (i + 1) / n_bins
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2 * 3)  # convert to km
            bin_means.append(overall[mask].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, bin_means, "b-", linewidth=2)
    ax.set_xlabel("Distance from Jumbo Statue (km)", fontsize=12)
    ax.set_ylabel("Mean Saliency", fontsize=12)
    ax.set_title("Saliency Decay with Distance", fontsize=14)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Jumbo location")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_distance_decay.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
