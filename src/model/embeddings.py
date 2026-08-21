"""
Spica AI - Token + Positional Embeddings

Converts token ids into dense vectors the transformer can process, and adds
positional information since self-attention has no built-in sense of order.
"""

import torch
import torch.nn as nn


class SpicaEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, seq_len) token ids
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        positions = torch.arange(T, device=idx.device)  # (T,)

        tok_emb = self.token_embedding(idx)          # (B, T, n_embd)
        pos_emb = self.position_embedding(positions)  # (T, n_embd), broadcasts over batch

        return self.dropout(tok_emb + pos_emb)  # (B, T, n_embd)
