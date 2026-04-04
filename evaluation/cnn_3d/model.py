"""
Evaluation wrapper for the CNN3D model.

Multi-frame model: requires 4 consecutive frames with temporal dimension.
Exposes get_model(metadata) and N_FRAMES=4 for evaluate_all.py.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn_3d import CNN3D

CHECKPOINT_PATH = ROOT / "runs" / "cnn_3d" / "checkpoints" / "best.pt"
N_FRAMES = 4
STACK_MODE = "temporal"


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
        # x: (B, k, H, W, C) -> (B, k, C, H, W) for CNN3D
        B, k, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3)

        if self.input_mean is not None:
            # Normalize each frame: input_mean (C,1,1) -> (1,1,C,1,1) for (B,k,C,H,W)
            mean = self.input_mean.unsqueeze(0).unsqueeze(0)
            std = self.input_std.unsqueeze(0).unsqueeze(0)
            x = (x - mean) / (std + 1e-7)
        pred_normalized = self.model(x)
        if self.input_mean is not None:
            pred = pred_normalized * self.target_std + self.target_mean
        else:
            pred = pred_normalized
        return pred


def get_model(metadata):
    n_vars = metadata["n_vars"]
    model = CNN3D(n_input_channels=n_vars, n_targets=6, n_frames=N_FRAMES)

    norm_stats = None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_stats = ckpt.get("norm_stats")
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}, using random weights")

    return EvalWrapper(model, norm_stats)
