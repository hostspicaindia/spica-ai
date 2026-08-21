"""
Spica AI - Training Loop

Ties everything together: data, model, optimizer, LR schedule, gradient
clipping, periodic validation, checkpointing.

Usage:
    python -m src.training.trainer
    python -m src.training.trainer --resume checkpoints/model_1m/latest.pt
"""

import argparse
import time
from pathlib import Path

import torch

from src.config.config_loader import load_config
from src.data.dataset import get_dataloader
from src.model.gpt import GPT
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import get_lr
from src.utils.logger import get_logger

logger = get_logger("trainer")

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_model(model_cfg) -> GPT:
    return GPT(
        vocab_size=model_cfg.vocab_size,
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_embd=model_cfg.n_embd,
        n_head=model_cfg.n_head,
        dropout=model_cfg.dropout,
        bias=model_cfg.bias,
        tied_embeddings=model_cfg.tied_embeddings,
    )


@torch.no_grad()
def estimate_val_loss(model, val_loader, eval_iters: int, device: str) -> float:
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_1m.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume training from")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    model = build_model(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg.learning_rate, train_cfg.weight_decay)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, map_location=device)
        logger.info(f"resumed from {args.resume} at step {start_step}")

    train_loader = get_dataloader("train", model_cfg.block_size, train_cfg.batch_size, shuffle=True)
    val_loader = get_dataloader("val", model_cfg.block_size, train_cfg.batch_size, shuffle=True)
    train_iter = iter(train_loader)

    checkpoint_dir = ROOT_DIR / train_cfg.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.learning_rate, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            logger.info(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s elapsed")

        if step > 0 and step % train_cfg.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, train_cfg.eval_iters, device)
            logger.info(f"step {step:5d} | val_loss {val_loss:.4f}")

        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / f"step_{step}.pt", model, optimizer, step, model_cfg)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, model_cfg)
            logger.info(f"checkpoint saved at step {step}")

    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, train_cfg.max_steps, model_cfg)
    logger.info("training complete, final checkpoint saved")


if __name__ == "__main__":
    main()
