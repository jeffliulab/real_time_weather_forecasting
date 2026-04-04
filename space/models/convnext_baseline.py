"""
ConvNeXt-Tiny baseline for weather forecasting.

Uses torchvision's ConvNeXt-Tiny with modified input/output layers
to accept 42-channel weather data and predict 6 target variables.
Trained from scratch (no pretrained weights).

Input:  (B, 42, 450, 449)
Output: (B, 6)
"""

import torch.nn as nn
from torchvision.models import convnext_tiny


class ConvNeXtBaseline(nn.Module):
    """
    ConvNeXt-Tiny adapted for weather forecasting.

    Modifications from ImageNet ConvNeXt-Tiny:
    - Stem conv: 3 → 42 input channels
    - Classifier head: 1000 → n_targets outputs
    - No pretrained weights (trained from scratch)
    """

    def __init__(self, n_input_channels=42, n_targets=6, **kwargs):
        super().__init__()
        model = convnext_tiny(weights=None)

        # Replace stem conv: 3ch → 42ch
        # ConvNeXt stem: features[0][0] is Conv2d(3, 96, 4, stride=4)
        model.features[0][0] = nn.Conv2d(
            n_input_channels, 96, kernel_size=4, stride=4
        )

        # Replace classifier head: 1000 → n_targets
        # ConvNeXt head: classifier = Sequential(LayerNorm, Flatten, Linear(768, 1000))
        model.classifier[2] = nn.Linear(768, n_targets)

        self.model = model

    def forward(self, x):
        return self.model(x)
