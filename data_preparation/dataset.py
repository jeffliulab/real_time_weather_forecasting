"""
Weather forecasting dataset — loads HRRR snapshots and pairs with 24h-ahead targets.

Supports single-frame and multi-frame loading for different model architectures.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


class WeatherDataset(Dataset):
    """
    Loads spatial weather snapshots and pairs them with 24h-ahead targets.

    For single-frame models: returns (C, H, W) input tensor.
    For multi-frame models: returns (k*C, H, W) or (k, C, H, W) stacked tensor.
    """

    def __init__(self, data_root, years, n_frames=1, stack_mode="channel",
                 normalize=True, norm_stats=None):
        """
        Args:
            data_root: path containing dataset/ symlink
            years: list of years to include (e.g., [2018, 2019])
            n_frames: number of consecutive frames (1 = single-frame)
            stack_mode: "channel" stacks along channel dim (for multi-frame 2D CNN),
                        "temporal" keeps separate time dim (for 3D CNN)
            normalize: whether to normalize inputs and targets
            norm_stats: dict with keys 'input_mean', 'input_std', 'target_mean', 'target_std'
                        If None and normalize=True, stats must be set later via set_norm_stats()
        """
        self.data_root = Path(data_root)
        self.dataset_dir = self.data_root / "dataset"
        self.n_frames = n_frames
        self.stack_mode = stack_mode
        self.normalize = normalize
        self.norm_stats = norm_stats

        targets_data = torch.load(self.dataset_dir / "targets.pt", weights_only=False)
        self.times = targets_data["time"]
        self.target_values = targets_data["values"]       # (T, 6)
        self.binary_labels = targets_data["binary_label"]  # (T,) bool
        self.target_vars = list(targets_data["variable_names"])

        metadata = torch.load(self.dataset_dir / "metadata.pt", weights_only=False)
        self.n_vars = metadata["n_vars"]
        self.variable_names = list(metadata["variable_names"])

        self._build_index(years)

    def _build_index(self, years):
        """Build list of valid (input_time_index, target_time_index) pairs."""
        times_years = self.times.astype("datetime64[Y]").astype(int) + 1970

        year_mask = np.isin(times_years, years)
        candidate_indices = np.where(year_mask)[0]

        self.samples = []
        for t_idx in candidate_indices:
            t24_idx = t_idx + 24
            if t24_idx >= len(self.times):
                continue
            first_frame_idx = t_idx - (self.n_frames - 1)
            if first_frame_idx < 0:
                continue
            self.samples.append((t_idx, t24_idx))

    def _get_input_path(self, t_idx):
        dt = pd.Timestamp(self.times[t_idx])
        return (self.dataset_dir / "inputs" / str(dt.year)
                / f"X_{dt.strftime('%Y%m%d%H')}.pt")

    def _load_frame(self, t_idx):
        """Load a single frame, return (C, H, W) float32 or None if NaN."""
        path = self._get_input_path(t_idx)
        if not path.exists():
            return None
        x = torch.load(path, weights_only=True).float()  # (H, W, C)
        if torch.isnan(x).any():
            return None
        return x.permute(2, 0, 1)  # (C, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_idx, t24_idx = self.samples[idx]

        if self.n_frames == 1:
            x = self._load_frame(t_idx)
            if x is None:
                return None
        else:
            frames = []
            for offset in range(-(self.n_frames - 1), 1):
                frame = self._load_frame(t_idx + offset)
                if frame is None:
                    return None
                frames.append(frame)

            if self.stack_mode == "channel":
                x = torch.cat(frames, dim=0)     # (k*C, H, W)
            else:
                x = torch.stack(frames, dim=0)   # (k, C, H, W)

        target = self.target_values[t24_idx]        # (6,)
        binary = self.binary_labels[t24_idx].float() # scalar

        if self.normalize and self.norm_stats is not None:
            x = (x - self.norm_stats["input_mean"]) / (self.norm_stats["input_std"] + 1e-7)
            target = (target - self.norm_stats["target_mean"]) / (self.norm_stats["target_std"] + 1e-7)

        return x, target, binary

    def set_norm_stats(self, stats):
        self.norm_stats = stats


def collate_skip_none(batch):
    """Custom collate that filters out None samples (NaN inputs)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def compute_norm_stats(data_root, years, n_samples=1000, seed=42):
    """
    Compute per-channel input mean/std and target mean/std from a subsample.
    Returns dict with tensors shaped for broadcasting.
    """
    ds = WeatherDataset(data_root, years, n_frames=1, normalize=False)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    input_sum = None
    input_sq_sum = None
    target_sum = None
    target_sq_sum = None
    count = 0

    for i in indices:
        sample = ds[i]
        if sample is None:
            continue
        x, target, _ = sample

        if input_sum is None:
            C = x.shape[0]
            input_sum = torch.zeros(C)
            input_sq_sum = torch.zeros(C)
            target_sum = torch.zeros(6)
            target_sq_sum = torch.zeros(6)

        # Per-channel stats: mean over spatial dims
        input_sum += x.mean(dim=(1, 2))
        input_sq_sum += (x ** 2).mean(dim=(1, 2))
        target_sum += target
        target_sq_sum += target ** 2
        count += 1

    input_mean = input_sum / count
    input_std = torch.sqrt(input_sq_sum / count - input_mean ** 2)
    target_mean = target_sum / count
    target_std = torch.sqrt(target_sq_sum / count - target_mean ** 2)

    return {
        "input_mean": input_mean.reshape(-1, 1, 1),     # (C, 1, 1)
        "input_std": input_std.reshape(-1, 1, 1),
        "target_mean": target_mean,                       # (6,)
        "target_std": target_std,
    }


def get_dataloaders(data_root, batch_size=8, n_frames=1, stack_mode="channel",
                    num_workers=4, train_years=None, val_years=None):
    """
    Build train and validation DataLoaders with normalization.

    Returns: train_loader, val_loader, norm_stats
    """
    if train_years is None:
        train_years = [2018, 2019]
    if val_years is None:
        val_years = [2020]

    stats_path = Path(data_root) / "norm_stats.pt"
    if stats_path.exists():
        norm_stats = torch.load(stats_path, weights_only=True)
        # Adjust for multi-frame channel stacking
        if n_frames > 1 and stack_mode == "channel":
            norm_stats = {
                "input_mean": norm_stats["input_mean"].repeat(n_frames, 1, 1),
                "input_std": norm_stats["input_std"].repeat(n_frames, 1, 1),
                "target_mean": norm_stats["target_mean"],
                "target_std": norm_stats["target_std"],
            }
    else:
        print("Computing normalization statistics (this may take a few minutes)...")
        norm_stats = compute_norm_stats(data_root, train_years)
        torch.save(norm_stats, stats_path)
        print(f"Saved normalization stats to {stats_path}")
        if n_frames > 1 and stack_mode == "channel":
            norm_stats = {
                "input_mean": norm_stats["input_mean"].repeat(n_frames, 1, 1),
                "input_std": norm_stats["input_std"].repeat(n_frames, 1, 1),
                "target_mean": norm_stats["target_mean"],
                "target_std": norm_stats["target_std"],
            }

    train_ds = WeatherDataset(data_root, train_years, n_frames=n_frames,
                              stack_mode=stack_mode, norm_stats=norm_stats)
    val_ds = WeatherDataset(data_root, val_years, n_frames=n_frames,
                            stack_mode=stack_mode, norm_stats=norm_stats)

    loader_kwargs = dict(collate_fn=collate_skip_none, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, **loader_kwargs)

    return train_loader, val_loader, norm_stats
