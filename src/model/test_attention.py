"""
Spica AI - Attention Sanity Check

Verifies output shape, and (more important) that causal masking actually
works: changing a future token must not change any earlier position's
output.

Usage:
    python -m src.model.test_attention
"""

import torch

from src.config.config_loader import load_config
from src.model.attention import CausalSelfAttention
from src.utils.logger import get_logger

logger = get_logger("test_attention")


def main():
    cfg = load_config("configs/model_1m.yaml")

    attn = CausalSelfAttention(
        n_embd=cfg.n_embd, n_head=cfg.n_head,
        block_size=cfg.block_size, dropout=0.0, bias=cfg.bias,
    )
    attn.eval()  # dropout off for a deterministic test

    B, T = 2, 16
    x = torch.randn(B, T, cfg.n_embd)

    with torch.no_grad():
        out = attn(x)

    shape_ok = tuple(out.shape) == (B, T, cfg.n_embd)
    logger.info(f"output shape: {tuple(out.shape)} (expected {(B, T, cfg.n_embd)})")
    logger.info(f"shape check: {'PASS' if shape_ok else 'FAIL'}")

    # causality check: change only the LAST token, earlier outputs must stay identical
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, cfg.n_embd)

    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x2)

    earlier_unchanged = torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6)
    logger.info(
        f"causality check (earlier positions unaffected by future token): "
        f"{'PASS' if earlier_unchanged else 'FAIL'}"
    )

    n_params = sum(p.numel() for p in attn.parameters())
    logger.info(f"attention params: {n_params:,}")


if __name__ == "__main__":
    main()
