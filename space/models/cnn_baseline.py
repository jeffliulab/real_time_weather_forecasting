"""
Baseline 2D CNN for single-frame weather forecasting.

Takes a single spatial snapshot (C, H, W) and predicts 6 weather variables 24h ahead.
Architecture: progressive downsampling with residual blocks, global average pooling, FC head.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two conv layers and optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class BaselineCNN(nn.Module):
    """
    Single-frame 2D CNN encoder for weather forecasting.

    Input:  (B, C, 450, 449) where C = n_input_channels (42)
    Output: (B, 6) — predicted weather variables
    """

    def __init__(self, n_input_channels=42, n_targets=6, base_channels=64):
        super().__init__()

        ch = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        # 450x449 -> 225x225

        self.layer1 = ResBlock(ch, ch, stride=1)          # 225x225
        self.layer2 = ResBlock(ch, ch * 2, stride=2)      # 113x113
        self.layer3 = ResBlock(ch * 2, ch * 4, stride=2)  # 57x57
        self.layer4 = ResBlock(ch * 4, ch * 4, stride=2)  # 29x29
        self.layer5 = ResBlock(ch * 4, ch * 8, stride=2)  # 15x15
        self.layer6 = ResBlock(ch * 8, ch * 8, stride=2)  # 8x8

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 8, ch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(ch * 2, n_targets),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x)
        return self.head(x)
