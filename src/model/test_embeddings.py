"""
Spica AI - Embeddings Sanity Check

Loads model_1m.yaml, builds SpicaEmbeddings, runs a dummy batch through it,
and checks the output shape is correct.

Usage:
    python -m src.model.test_embeddings
"""

import torch

from src.config.config_loader import load_config
from src.model.embeddings import SpicaEmbeddings
from src.utils.logger import get_logger

logger = get_logger("test_embeddings")


def main():
    cfg = load_config("configs/model_1m.yaml")

    emb = SpicaEmbeddings(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )

    batch_size, seq_len = 4, 32
    dummy_idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    out = emb(dummy_idx)

    expected_shape = (batch_size, seq_len, cfg.n_embd)
    passed = tuple(out.shape) == expected_shape

    logger.info(f"input shape : {tuple(dummy_idx.shape)}")
    logger.info(f"output shape: {tuple(out.shape)} (expected {expected_shape})")
    logger.info("PASS" if passed else "FAIL")

    n_params = sum(p.numel() for p in emb.parameters())
    logger.info(f"embedding params (token+position): {n_params:,}")


if __name__ == "__main__":
    main()
