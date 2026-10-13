from __future__ import annotations

import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None
        # Multiplicative per-head mask used for ablation (1 = keep, 0 = remove).
        self.register_buffer("head_mask", torch.ones(n_heads), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        attention = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        self.last_attention = attention.detach()
        z = self.dropout(attention) @ v
        z = z * self.head_mask.view(1, -1, 1, 1)
        z = z.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(z)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        if t > self.max_length:
            raise ValueError(f"length {t} exceeds max_length {self.max_length}")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def attention_maps(self) -> list[torch.Tensor]:
        return [block.attn.last_attention for block in self.blocks]

