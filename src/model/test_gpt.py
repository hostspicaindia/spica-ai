"""
Spica AI - Full GPT Model Sanity Check

Usage:
    python -m src.model.test_gpt
"""

import math

import torch

from src.config.config_loader import load_config
from src.model.gpt import GPT
from src.utils.logger import get_logger

logger = get_logger("test_gpt")


def main():
    cfg = load_config("configs/model_1m.yaml")

    model = GPT(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        bias=cfg.bias,
        tied_embeddings=cfg.tied_embeddings,
    )
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets)

    shape_ok = tuple(logits.shape) == (B, T, cfg.vocab_size)
    logger.info(f"logits shape: {tuple(logits.shape)} (expected {(B, T, cfg.vocab_size)})")
    logger.info(f"shape check: {'PASS' if shape_ok else 'FAIL'}")
    logger.info(f"loss: {loss.item():.4f}  finite: {torch.isfinite(loss).item()}")

    # A freshly-initialized model predicts ~uniformly over the vocab, so
    # loss should start near ln(vocab_size) - a classic first sanity check.
    expected_random_loss = math.log(cfg.vocab_size)
    logger.info(f"expected ~random-init loss: {expected_random_loss:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"total params: {total_params:,}")

    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    gen_ok = generated.size(1) == prompt.size(1) + 10
    logger.info(
        f"generate(): input_len={prompt.size(1)} -> output_len={generated.size(1)} "
        f"({'PASS' if gen_ok else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
