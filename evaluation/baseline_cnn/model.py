"""
Evaluation wrapper for the baseline CNN model.

This file exposes get_model(metadata) as required by evaluate.py.
The model's forward() accepts (B, 450, 449, c) and returns (B, 6).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent.parent  # assignment2/
sys.path.insert(0, str(ROOT))

from models.cnn_baseline import BaselineCNN

CHECKPOINT_PATH = ROOT / "runs" / "cnn_baseline" / "checkpoints" / "best.pt"

TARGET_VARS = [
    "TMP@2m_above_ground", "RH@2m_above_ground",
    "UGRD@10m_above_ground", "VGRD@10m_above_ground",
    "GUST@surface", "APCP_1hr_acc_fcst@surface",
]


class EvalWrapper(nn.Module):
    """
    Wraps the trained CNN to match evaluate.py's interface.

    evaluate.py calls forward(x) with x of shape (B, 450, 449, c) in float32.
    Our CNN expects (B, c, 450, 449) with normalization applied.
    """

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
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        if self.input_mean is not None:
            x = (x - self.input_mean) / (self.input_std + 1e-7)

        pred_normalized = self.model(x)  # (B, 6)

        if self.input_mean is not None:
            pred = pred_normalized * self.target_std + self.target_mean
        else:
            pred = pred_normalized

        return pred


def get_model(metadata):
    n_vars = metadata["n_vars"]
    model = BaselineCNN(n_input_channels=n_vars, n_targets=6)

    norm_stats = None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_stats = ckpt.get("norm_stats")
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}, using random weights")

    return EvalWrapper(model, norm_stats)
