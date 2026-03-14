"""Quick smoke test: verify all models can be imported and run with small inputs."""

import torch
from models import create_model, MODEL_REGISTRY

print("Available models:", list(MODEL_REGISTRY.keys()))
for name in MODEL_REGISTRY:
    m = create_model(name, n_input_channels=42)
    n = sum(p.numel() for p in m.parameters())
    print(f"  {name}: {n:,} params")

# Use small spatial dims to avoid OOM on login/interactive nodes
H, W = 30, 30

m1 = create_model("cnn_baseline", n_input_channels=42)
m1.eval()
with torch.no_grad():
    print("\ncnn_baseline forward:", m1(torch.randn(1, 42, H, W)).shape)

m2 = create_model("cnn_multi_frame", n_input_channels=42, n_frames=4)
m2.eval()
with torch.no_grad():
    print("cnn_multi_frame forward:", m2(torch.randn(1, 42 * 4, H, W)).shape)

m3 = create_model("cnn_3d", n_input_channels=42, n_frames=4)
m3.eval()
with torch.no_grad():
    print("cnn_3d forward:", m3(torch.randn(1, 4, 42, H, W)).shape)

m4 = create_model("vit", n_input_channels=42, patch_size=15)
m4.eval()
with torch.no_grad():
    # ViT requires 450x450 input (hardcoded positional embeddings)
    print("vit forward:", m4(torch.randn(1, 42, 450, 449)).shape)

# Test dataset import
from data_preparation.dataset import WeatherDataset
print("\nDataset class imported successfully")

print("\nAll smoke tests passed!")
