"""
model.py — CNN + Multi-Head Self-Attention Classifier for AI Voice Detection
=============================================================================
Architecture:
  1. Input: 768-dim Wav2Vec2 features (pre-extracted)
  2. Reshape → 1D "image" for CNN
  3. Multi-scale Conv1D feature extraction
  4. Multi-Head Self-Attention over temporal patches
  5. Global attention pooling
  6. Regularized MLP head → 2-class output (Real / Fake)

Why this beats a plain Linear classifier:
  - CNN captures local spectral patterns
  - Attention weights which patches matter most
  - Dropout + BatchNorm + weight_decay fight overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention for 1D feature sequences."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, T, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class ConvBlock(nn.Module):
    """Conv1D → BN → ReLU → optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)) + self.res(x))


class CNNAttentionClassifier(nn.Module):
    """
    CNN + Multi-Head Attention classifier.

    Input shape:  (B, 768)   — mean-pooled Wav2Vec2 features
    Output shape: (B, 2)     — logits for [Real, Fake]

    Internal flow:
      (B, 768) → unsqueeze → (B, 1, 768) [treat 768 as "time" axis with 1 channel]
      → 3× ConvBlock (1→32→64→128 channels, progressively halve length)
      → (B, 128, 96) → permute → (B, 96, 128) [sequence of 96 patch tokens]
      → LayerNorm
      → MultiHeadSelfAttention(128, heads=4)
      → LayerNorm (post)
      → Global Attention Pooling (learnable query)
      → (B, 128) → MLP head → (B, 2)
    """

    def __init__(self, feat_dim: int = 768, num_classes: int = 2,
                 num_heads: int = 4, mlp_dropout: float = 0.4):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────
        self.cnn = nn.Sequential(
            ConvBlock(1,  32,  kernel_size=7),
            nn.MaxPool1d(2),          # 768 → 384
            ConvBlock(32, 64,  kernel_size=5),
            nn.MaxPool1d(2),          # 384 → 192
            ConvBlock(64, 128, kernel_size=3),
            nn.MaxPool1d(2),          # 192 → 96
        )

        self.token_dim = 128

        # ── Transformer-style attention ───────────────────────────────
        self.norm1   = nn.LayerNorm(self.token_dim)
        self.attn    = MultiHeadSelfAttention(self.token_dim, num_heads=num_heads)
        self.norm2   = nn.LayerNorm(self.token_dim)

        # Learnable "class token" for global pooling via cross-attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.token_dim))

        # ── MLP head ──────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(self.token_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(mlp_dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 768)
        B = x.size(0)

        # CNN
        x = x.unsqueeze(1)                     # (B, 1, 768)
        x = self.cnn(x)                        # (B, 128, 96)
        x = x.permute(0, 2, 1)                 # (B, 96, 128)

        # Pre-norm self-attention (residual)
        x = x + self.attn(self.norm1(x))
        x = self.norm2(x)

        # Global pooling: mean over sequence
        x = x.mean(dim=1)                      # (B, 128)

        return self.head(x)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = CNNAttentionClassifier()
    dummy = torch.randn(4, 768)
    out = m(dummy)
    print("Output shape:", out.shape)   # expect (4, 2)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total params: {total_params:,}")
