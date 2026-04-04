"""
ResNet-18 baseline for weather forecasting.

Uses torchvision's ResNet-18 with modified input/output layers
to accept 42-channel weather data and predict 6 target variables.
Trained from scratch (no pretrained weights).

Input:  (B, 42, 450, 449)
Output: (B, 6)
"""

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Baseline(nn.Module):
    """
    Standard ResNet-18 adapted for weather forecasting.

    Modifications from ImageNet ResNet-18:
    - conv1: 3 → 42 input channels
    - fc: 1000 → n_targets output classes
    - No pretrained weights (trained from scratch)
    """

    def __init__(self, n_input_channels=42, n_targets=6, **kwargs):
        super().__init__()
        model = resnet18(weights=None)

        # Replace first conv: 3ch → 42ch
        model.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head: 1000 → n_targets
        model.fc = nn.Linear(512, n_targets)

        self.model = model

    def forward(self, x):
        return self.model(x)
