"""
Spica AI - Transformer Block Sanity Check

Usage:
    python -m src.model.test_block
"""

import torch

from src.config.config_loader import load_config
from src.model.block import Block
from src.utils.logger import get_logger

logger = get_logger("test_block")


def main():
    cfg = load_config("configs/model_1m.yaml")

    block = Block(
        n_embd=cfg.n_embd, n_head=cfg.n_head,
        block_size=cfg.block_size, dropout=0.0, bias=cfg.bias,
    )
    block.eval()

    B, T = 2, 16
    x = torch.randn(B, T, cfg.n_embd)

    with torch.no_grad():
        out = block(x)

    shape_ok = tuple(out.shape) == (B, T, cfg.n_embd)
    logger.info(f"output shape: {tuple(out.shape)} (expected {(B, T, cfg.n_embd)})")
    logger.info(f"shape check: {'PASS' if shape_ok else 'FAIL'}")

    n_params = sum(p.numel() for p in block.parameters())
    logger.info(f"block params: {n_params:,} (should be ~= attention + feedforward + 2 layernorms)")


if __name__ == "__main__":
    main()
