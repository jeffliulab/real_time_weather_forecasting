"""
Real-time weather forecast inference pipeline (Part 4).

Fetches the latest GFS analysis data from NOAA, runs the trained CNN model,
and outputs a 24-hour weather forecast for the Jumbo Statue at Tufts University.

Usage:
    python -m inference.predict --checkpoint runs/cnn_baseline/checkpoints/best.pt

TODO:
    - [ ] Implement fetch_gfs_data() to download latest GFS 0.25-deg analysis
    - [ ] Implement regrid_to_hrrr() to interpolate GFS onto the HRRR 3km grid
    - [ ] Map GFS variables to the 42-channel input format (VAR_LEVELS order)
    - [ ] Handle missing variables (GFS has different variable set than HRRR)
    - [ ] Add web UI or CLI output formatting
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import create_model, get_model_defaults


def fetch_gfs_data(timestamp=None):
    """
    Fetch GFS analysis data from NOAA for the New England region.

    GFS data is available at:
        https://nomads.ncep.noaa.gov/dods/gfs_0p25/

    Args:
        timestamp: datetime object for the analysis time.
                   If None, uses the latest available cycle.

    Returns:
        np.ndarray of shape (450, 449, 42) matching the HRRR input format,
        or None if data is not available.
    """
    raise NotImplementedError(
        "GFS data fetching not yet implemented. "
        "See NOAA NOMADS API: https://nomads.ncep.noaa.gov/"
    )


def load_model(checkpoint_path, device="cpu"):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model_name = args["model"]

    defaults = get_model_defaults(model_name)
    n_frames = args.get("n_frames") or defaults["n_frames"]

    model_kwargs = {
        "n_input_channels": 42,
        "n_targets": 6,
        "base_channels": args.get("base_channels", 64),
    }
    if n_frames > 1:
        model_kwargs["n_frames"] = n_frames

    model = create_model(model_name, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm_stats = ckpt.get("norm_stats")
    return model, norm_stats


TARGET_VARS = [
    ("TMP@2m_above_ground", "Temperature (2m)", "K"),
    ("RH@2m_above_ground", "Relative Humidity (2m)", "%"),
    ("UGRD@10m_above_ground", "U-Wind (10m)", "m/s"),
    ("VGRD@10m_above_ground", "V-Wind (10m)", "m/s"),
    ("GUST@surface", "Wind Gust", "m/s"),
    ("APCP_1hr_acc_fcst@surface", "Precipitation (1hr)", "mm"),
]


def predict(model, input_tensor, norm_stats, device="cpu"):
    """
    Run inference on a single input frame.

    Args:
        model: trained nn.Module
        input_tensor: (450, 449, 42) numpy array or torch tensor
        norm_stats: dict with input_mean, input_std, target_mean, target_std
        device: torch device

    Returns:
        dict mapping variable name to predicted value
    """
    if isinstance(input_tensor, np.ndarray):
        input_tensor = torch.from_numpy(input_tensor).float()

    x = input_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    if norm_stats:
        x = (x - norm_stats["input_mean"]) / (norm_stats["input_std"] + 1e-7)

    x = x.to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu()  # (6,)

    if norm_stats:
        pred = pred * norm_stats["target_std"] + norm_stats["target_mean"]

    results = {}
    for i, (var_name, display_name, unit) in enumerate(TARGET_VARS):
        results[var_name] = {
            "value": pred[i].item(),
            "display_name": display_name,
            "unit": unit,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Real-time weather forecast")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Loading model...")
    model, norm_stats = load_model(args.checkpoint, args.device)

    print("Fetching latest weather data...")
    try:
        input_data = fetch_gfs_data()
    except NotImplementedError as e:
        print(f"\n{e}")
        print("\nTo complete this pipeline, implement fetch_gfs_data() to:")
        print("  1. Download GFS 0.25-deg analysis from NOAA NOMADS")
        print("  2. Regrid to HRRR 3km Lambert Conformal grid (450x449)")
        print("  3. Map to 42-channel format defined in data/data_spec.py")
        return

    now = datetime.utcnow()
    forecast_time = now + timedelta(hours=24)

    results = predict(model, input_data, norm_stats, args.device)

    print(f"\n{'='*50}")
    print(f"  24h Weather Forecast for Tufts Jumbo Statue")
    print(f"  Based on: {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Valid at:  {forecast_time.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")
    for var_name, info in results.items():
        print(f"  {info['display_name']:30s} {info['value']:8.2f} {info['unit']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
