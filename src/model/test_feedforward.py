"""
Spica AI - Feed-Forward Sanity Check

Usage:
    python -m src.model.test_feedforward
"""

import torch

from src.config.config_loader import load_config
from src.model.feedforward import FeedForward
from src.utils.logger import get_logger

logger = get_logger("test_feedforward")


def main():
    cfg = load_config("configs/model_1m.yaml")

    ff = FeedForward(n_embd=cfg.n_embd, dropout=0.0, bias=cfg.bias)
    ff.eval()

    B, T = 2, 16
    x = torch.randn(B, T, cfg.n_embd)

    with torch.no_grad():
        out = ff(x)

    shape_ok = tuple(out.shape) == (B, T, cfg.n_embd)
    logger.info(f"output shape: {tuple(out.shape)} (expected {(B, T, cfg.n_embd)})")
    logger.info(f"shape check: {'PASS' if shape_ok else 'FAIL'}")

    n_params = sum(p.numel() for p in ff.parameters())
    logger.info(f"feedforward params: {n_params:,}")


if __name__ == "__main__":
    main()
