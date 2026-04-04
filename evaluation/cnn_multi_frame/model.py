"""
Evaluation wrapper for the MultiFrameCNN model.

Multi-frame model: requires 4 consecutive frames stacked along channels.
Exposes get_model(metadata) and N_FRAMES=4 for evaluate_all.py.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

EVAL_DIR = Path(__file__).parent
ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(ROOT))

from models.cnn_multi_frame import MultiFrameCNN

CHECKPOINT_PATH = ROOT / "runs" / "cnn_multi_frame" / "checkpoints" / "best.pt"
N_FRAMES = 4
STACK_MODE = "channel"


class EvalWrapper(nn.Module):
    def __init__(self, model, norm_stats, n_frames):
        super().__init__()
        self.model = model
        self.n_frames = n_frames
        if norm_stats is not None:
            # norm_stats from checkpoint already have multi-frame repeat applied
            self.register_buffer("input_mean", norm_stats["input_mean"])
            self.register_buffer("input_std", norm_stats["input_std"])
            self.register_buffer("target_mean", norm_stats["target_mean"])
            self.register_buffer("target_std", norm_stats["target_std"])
        else:
            self.input_mean = None

    def forward(self, x):
        # x: (B, k, H, W, C) -> permute each frame and channel-stack
        B, k, H, W, C = x.shape
        # Permute to (B, k, C, H, W) then reshape to (B, k*C, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, k * C, H, W)

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
    model = MultiFrameCNN(n_input_channels=n_vars, n_targets=6, n_frames=N_FRAMES)

    norm_stats = None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_stats = ckpt.get("norm_stats")
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}, using random weights")

    return EvalWrapper(model, norm_stats, N_FRAMES)
