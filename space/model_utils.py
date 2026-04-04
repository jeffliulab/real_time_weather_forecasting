"""
Model loading and inference utilities for the weather forecast demo.

Wraps the existing inference/predict.py logic, adding user-friendly
post-processing (Celsius, wind speed/direction, rain likelihood).
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

# In HF Space, models/ is in the same directory as this file
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model, get_model_defaults

# ── Model cache (loaded once, reused across requests) ──────────────────
_model_cache: dict = {}


TARGET_VARS = [
    ("TMP@2m_above_ground",       "Temperature (2m)",    "K"),
    ("RH@2m_above_ground",        "Relative Humidity",   "%"),
    ("UGRD@10m_above_ground",     "U-Wind (10m)",        "m/s"),
    ("VGRD@10m_above_ground",     "V-Wind (10m)",        "m/s"),
    ("GUST@surface",              "Wind Gust",           "m/s"),
    ("APCP_1hr_acc_fcst@surface", "Precipitation (1hr)", "mm"),
]

# Available models with display info
AVAILABLE_MODELS = {
    "cnn_baseline": {
        "display_name": "CNN Baseline",
        "checkpoint": "checkpoints/cnn_baseline.pt",
        "params": "11.3M",
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "checkpoint": "checkpoints/resnet18.pt",
        "params": "11.2M",
    },
}


def load_model(model_name: str, device: str = "cpu"):
    """
    Load a trained model from checkpoint. Caches in memory for reuse.

    Returns:
        (model, norm_stats) tuple
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_path = PROJECT_ROOT / AVAILABLE_MODELS[model_name]["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    ckpt_model_name = args["model"]

    defaults = get_model_defaults(ckpt_model_name)
    n_frames = args.get("n_frames") or defaults["n_frames"]

    model_kwargs = {
        "n_input_channels": 42,
        "n_targets": 6,
        "base_channels": args.get("base_channels", 64),
    }
    if n_frames > 1:
        model_kwargs["n_frames"] = n_frames

    model = create_model(ckpt_model_name, **model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")

    _model_cache[model_name] = (model, norm_stats)
    return model, norm_stats


def predict_raw(model, norm_stats, input_array: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Run inference on a (450, 449, 42) input array.

    Returns:
        np.ndarray of shape (6,) with denormalized physical values.
    """
    x = torch.from_numpy(input_array).float()
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 42, 450, 449)

    if norm_stats:
        mean = norm_stats["input_mean"]
        std = norm_stats["input_std"]
        # Ensure correct device
        if isinstance(mean, torch.Tensor):
            mean = mean.float()
            std = std.float()
        x = (x - mean) / (std + 1e-7)

    x = x.to(device)
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu()  # (6,)

    if norm_stats:
        target_mean = norm_stats["target_mean"]
        target_std = norm_stats["target_std"]
        if isinstance(target_mean, torch.Tensor):
            target_mean = target_mean.float()
            target_std = target_std.float()
        pred = pred * target_std + target_mean

    return pred.numpy()


def _wind_direction_str(degrees: float) -> str:
    """Convert wind direction in degrees to compass string."""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(degrees / 22.5) % 16
    return dirs[idx]


def format_forecast(pred: np.ndarray) -> dict:
    """
    Convert raw model output (6 physical values) into a user-friendly forecast dict.
    """
    temp_k = float(pred[0])
    rh = float(pred[1])
    u_wind = float(pred[2])
    v_wind = float(pred[3])
    gust = float(pred[4])
    apcp = float(pred[5])

    # Derived quantities
    temp_c = temp_k - 273.15
    temp_f = temp_c * 9 / 5 + 32
    wind_speed = math.sqrt(u_wind**2 + v_wind**2)
    # Meteorological wind direction: direction FROM which wind blows
    wind_dir_deg = (math.degrees(math.atan2(-u_wind, -v_wind)) + 360) % 360
    wind_dir_str = _wind_direction_str(wind_dir_deg)

    # Rain likelihood based on APCP threshold
    apcp = max(apcp, 0.0)  # Clamp negative predictions
    if apcp > 5.0:
        rain_str = "Heavy Rain Likely"
    elif apcp > 2.0:
        rain_str = "Rain Likely"
    elif apcp > 0.5:
        rain_str = "Light Rain Possible"
    else:
        rain_str = "No Rain Expected"

    return {
        "temperature_k": temp_k,
        "temperature_c": temp_c,
        "temperature_f": temp_f,
        "humidity_pct": max(0.0, min(100.0, rh)),
        "u_wind_ms": u_wind,
        "v_wind_ms": v_wind,
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir_deg,
        "wind_dir_str": wind_dir_str,
        "gust_ms": max(gust, 0.0),
        "precipitation_mm": apcp,
        "rain_status": rain_str,
    }


def run_forecast(model_name: str, input_array: np.ndarray, device: str = "cpu") -> dict:
    """Full pipeline: load model → predict → format results."""
    model, norm_stats = load_model(model_name, device)
    pred = predict_raw(model, norm_stats, input_array, device)
    return format_forecast(pred)
