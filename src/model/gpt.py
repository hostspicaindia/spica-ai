"""
Spica AI - GPT-2 Style Decoder-Only Transformer

Assembles embeddings + a stack of transformer blocks + final LayerNorm +
language modeling head into the full model. This exact code is reused for
every size in the roadmap (1M -> 10M -> 100M -> 500M -> 1B); only the
configs/*.yaml changes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.block import Block
from src.model.embeddings import SpicaEmbeddings


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        tied_embeddings: bool = True,
    ):
        super().__init__()
        self.block_size = block_size

        self.embeddings = SpicaEmbeddings(vocab_size, n_embd, block_size, dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        if tied_embeddings:
            # reuse the token embedding matrix as the output projection -
            # standard GPT-2 trick, avoids paying for the huge vocab twice
            self.lm_head.weight = self.embeddings.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
