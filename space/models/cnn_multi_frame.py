"""
Multi-frame 2D CNN for weather forecasting.

Stacks k consecutive spatial snapshots along the channel dimension,
allowing the model to learn temporal patterns with standard 2D convolutions.

Input:  (B, k*C, H, W)
Output: (B, 6)
"""

import torch
import torch.nn as nn
from .cnn_baseline import ResBlock


class MultiFrameCNN(nn.Module):
    """
    Multi-frame 2D CNN that concatenates consecutive frames along channels.

    Identical backbone to BaselineCNN but with an adapted stem for k*C input channels,
    plus a temporal mixing layer after the stem.
    """

    def __init__(self, n_input_channels=42, n_targets=6, n_frames=4, base_channels=64):
        super().__init__()
        self.n_frames = n_frames
        in_ch = n_input_channels * n_frames
        ch = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch * 2, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock(ch, ch, stride=1)
        self.layer2 = ResBlock(ch, ch * 2, stride=2)
        self.layer3 = ResBlock(ch * 2, ch * 4, stride=2)
        self.layer4 = ResBlock(ch * 4, ch * 4, stride=2)
        self.layer5 = ResBlock(ch * 4, ch * 8, stride=2)
        self.layer6 = ResBlock(ch * 8, ch * 8, stride=2)

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
