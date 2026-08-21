"""
Spica AI - Dataset/DataLoader Sanity Check

Usage:
    python -m src.data.test_dataset
"""

import torch

from src.config.config_loader import load_config
from src.data.dataset import TokenDataset, get_dataloader
from src.tokenizer.qwen_tokenizer import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger("test_dataset")

BATCH_SIZE = 8  # small, just for this sanity check - real training config comes later


def main():
    cfg = load_config("configs/model_1m.yaml")

    train_ds = TokenDataset("train", block_size=cfg.block_size)
    val_ds = TokenDataset("val", block_size=cfg.block_size)
    logger.info(f"train dataset: {len(train_ds):,} possible chunks")
    logger.info(f"val dataset  : {len(val_ds):,} possible chunks")

    loader = get_dataloader("train", block_size=cfg.block_size, batch_size=BATCH_SIZE)
    x, y = next(iter(loader))

    shape_ok = tuple(x.shape) == (BATCH_SIZE, cfg.block_size) and x.shape == y.shape
    logger.info(f"x shape: {tuple(x.shape)}  y shape: {tuple(y.shape)} (expected {(BATCH_SIZE, cfg.block_size)})")
    logger.info(f"shape check: {'PASS' if shape_ok else 'FAIL'}")

    # y must be x shifted by exactly one position, within a sample
    shift_ok = torch.equal(x[:, 1:], y[:, :-1])
    logger.info(f"shift check (y = x shifted by 1): {'PASS' if shift_ok else 'FAIL'}")

    # human-readable check: decode one sample back to text
    tokenizer = load_tokenizer()
    sample_text = tokenizer.decode(x[0].tolist())
    logger.info(f"decoded sample x[0]: {sample_text!r}")


if __name__ == "__main__":
    main()
