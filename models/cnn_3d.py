"""
3D CNN for spatiotemporal weather forecasting.

Uses Conv3d to jointly model spatial and temporal patterns
from multiple consecutive weather snapshots.

Input:  (B, k, C, H, W) — k frames, each with C channels
Output: (B, 6)
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3D residual block with separate temporal and spatial convolutions."""

    def __init__(self, in_ch, out_ch, stride_spatial=1, stride_temporal=1):
        super().__init__()
        stride = (stride_temporal, stride_spatial, stride_spatial)

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if any(s != 1 for s in stride) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class CNN3D(nn.Module):
    """
    3D CNN for spatiotemporal weather forecasting.

    Input:  (B, k, C, H, W) where k=n_frames, C=n_input_channels
    Output: (B, 6)

    The temporal dimension is collapsed early (by stride-2 temporal convolutions
    and pooling), while spatial dimensions are progressively downsampled.
    """

    def __init__(self, n_input_channels=42, n_targets=6, n_frames=4, base_channels=64):
        super().__init__()
        ch = base_channels

        # (B, k, C, H, W) -> (B, C, k, H, W) is done in forward()
        self.stem = nn.Sequential(
            nn.Conv3d(n_input_channels, ch, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
        )
        # spatial: 450x449 -> 225x225, temporal: k -> k

        self.layer1 = ResBlock3D(ch, ch, stride_spatial=1, stride_temporal=1)
        # Collapse temporal dimension
        self.layer2 = ResBlock3D(ch, ch * 2, stride_spatial=2, stride_temporal=2)
        # temporal: k -> k//2, spatial: 225 -> 113
        self.layer3 = ResBlock3D(ch * 2, ch * 4, stride_spatial=2, stride_temporal=2)
        # temporal: k//2 -> 1 (for k=4), spatial: 113 -> 57
        self.layer4 = ResBlock3D(ch * 4, ch * 4, stride_spatial=2, stride_temporal=1)
        # spatial: 57 -> 29
        self.layer5 = ResBlock3D(ch * 4, ch * 8, stride_spatial=2, stride_temporal=1)
        # spatial: 29 -> 15
        self.layer6 = ResBlock3D(ch * 8, ch * 8, stride_spatial=2, stride_temporal=1)
        # spatial: 15 -> 8

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 8, ch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(ch * 2, n_targets),
        )

    def forward(self, x):
        # x: (B, k, C, H, W) -> (B, C, k, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x)
        return self.head(x)
