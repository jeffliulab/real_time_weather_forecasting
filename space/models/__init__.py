"""
Model registry for weather forecasting architectures.

Usage:
    from models import create_model, MODEL_REGISTRY
    model = create_model("cnn_baseline", n_input_channels=42)
"""

from .cnn_baseline import BaselineCNN
from .cnn_multi_frame import MultiFrameCNN
from .cnn_3d import CNN3D
from .vit import WeatherViT
from .resnet_baseline import ResNet18Baseline
from .convnext_baseline import ConvNeXtBaseline

MODEL_REGISTRY = {
    "cnn_baseline": BaselineCNN,
    "cnn_multi_frame": MultiFrameCNN,
    "cnn_3d": CNN3D,
    "vit": WeatherViT,
    "resnet18": ResNet18Baseline,
    "convnext_tiny": ConvNeXtBaseline,
}

# Default model-specific settings
MODEL_DEFAULTS = {
    "cnn_baseline": {"n_frames": 1, "stack_mode": "channel"},
    "cnn_multi_frame": {"n_frames": 4, "stack_mode": "channel"},
    "cnn_3d": {"n_frames": 4, "stack_mode": "temporal"},
    "vit": {"n_frames": 1, "stack_mode": "channel"},
    "resnet18": {"n_frames": 1, "stack_mode": "channel"},
    "convnext_tiny": {"n_frames": 1, "stack_mode": "channel"},
}


def create_model(name, **kwargs):
    """Instantiate a model by name with given kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_model_defaults(name):
    """Return default n_frames and stack_mode for a model."""
    return MODEL_DEFAULTS.get(name, {"n_frames": 1, "stack_mode": "channel"})
