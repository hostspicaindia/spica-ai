"""
Spica AI - Multi-Head Causal Self-Attention

Lets each token gather context from itself and earlier tokens in the
sequence. "Causal" means a token can never attend to future tokens -
required since Spica AI generates text left to right, one token at a time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        qkv = self.qkv_proj(x)               # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # split into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # is_causal=True applies the future-masking automatically
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, n_head, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back

        return self.resid_dropout(self.out_proj(out))
