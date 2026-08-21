"""
Spica AI - Position-wise Feed-Forward Network

Runs after attention in each transformer block. Attention mixes information
BETWEEN tokens; this layer processes EACH token independently through a
small expand-then-contract neural net, adding representational capacity.
GPT-2 style: n_embd -> 4*n_embd -> GELU -> n_embd.
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
