"""
Spica AI - Transformer Block

One block = attention + feedforward, each wrapped in a residual connection
and preceded by LayerNorm (pre-norm, the modern GPT-2-style convention).
Stacking n_layer of these is the entire transformer body.
"""

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # residual around attention
        x = x + self.ff(self.ln2(x))    # residual around feedforward
        return x
