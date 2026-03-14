"""
Vision Transformer (ViT) for weather forecasting.

Splits the spatial input into non-overlapping patches, projects each patch
into an embedding, and processes them through a Transformer encoder.

Input:  (B, C, H, W) — single frame with C channels
Output: (B, 6)
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert spatial input into a sequence of patch embeddings."""

    def __init__(self, in_channels, embed_dim, patch_size, img_h, img_w):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_h = img_h // patch_size
        self.n_patches_w = img_w // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, nH, nW) -> (B, n_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class WeatherViT(nn.Module):
    """
    Vision Transformer for weather forecasting.

    Input:  (B, C, 450, 449) — pads width to 450 internally
    Output: (B, 6)

    Patches the input into 15x15 patches (30x30 = 900 tokens),
    adds CLS token and positional embeddings, runs through Transformer,
    and predicts from the CLS token.
    """

    def __init__(self, n_input_channels=42, n_targets=6, patch_size=15,
                 embed_dim=256, n_layers=6, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        img_h, img_w = 450, 450  # pad to square

        self.patch_embed = PatchEmbedding(n_input_channels, embed_dim,
                                          patch_size, img_h, img_w)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, n_targets),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pad width from 449 to 450 if needed
        if W < 450:
            x = nn.functional.pad(x, (0, 450 - W))

        patches = self.patch_embed(x)                       # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, D)
        x = torch.cat([cls, patches], dim=1)                # (B, n_patches+1, D)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        cls_out = x[:, 0]                                   # (B, D)
        return self.head(cls_out)
