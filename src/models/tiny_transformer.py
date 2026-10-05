"""Tiny Transformer model in PyTorch.

Small enough to reason about and trace by hand, yet architecturally complete
(Multi-Head Self-Attention, LayerNorm, Residual Connections, FFN with GELU/ReLU).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Individual linear projections for explicit tracing
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, S)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)  # (B, H, S, d_k)

        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 64, d_ff: int = 128, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.activation = activation
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        if self.activation == "gelu":
            h = F.gelu(h)
        else:
            h = F.relu(h)
        out = self.linear2(h)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 128, activation: str = "gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        norm1 = self.ln1(x)
        attn_out = self.attn(norm1)
        x = x + attn_out

        norm2 = self.ln2(x)
        ffn_out = self.ffn(norm2)
        x = x + ffn_out
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        vocab_size: int = 1000,
        max_seq_len: int = 64,
        num_classes: int = 10,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, activation=activation)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be embedding/hidden states of shape (B, S, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        # Average pooling over sequence dimension
        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        return logits
