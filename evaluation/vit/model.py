"""
Evaluation wrapper for the WeatherViT model.

Exposes get_model(metadata) as required by evaluate_all.py.
The model's forward() accepts (B, 450, 449, c) and returns (B, 6).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from models.vit import WeatherViT

CHECKPOINT_PATH = ROOT / "runs" / "vit" / "checkpoints" / "best.pt"


class EvalWrapper(nn.Module):
    def __init__(self, model, norm_stats):
        super().__init__()
        self.model = model
        if norm_stats is not None:
            self.register_buffer("input_mean", norm_stats["input_mean"])
            self.register_buffer("input_std", norm_stats["input_std"])
            self.register_buffer("target_mean", norm_stats["target_mean"])
            self.register_buffer("target_std", norm_stats["target_std"])
        else:
            self.input_mean = None

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        if self.input_mean is not None:
            x = (x - self.input_mean) / (self.input_std + 1e-7)
        pred_normalized = self.model(x)
        if self.input_mean is not None:
            pred = pred_normalized * self.target_std + self.target_mean
        else:
            pred = pred_normalized
        return pred


def get_model(metadata):
    n_vars = metadata["n_vars"]
    model = WeatherViT(n_input_channels=n_vars, n_targets=6)

    norm_stats = None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_stats = ckpt.get("norm_stats")
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}, using random weights")

    return EvalWrapper(model, norm_stats)
