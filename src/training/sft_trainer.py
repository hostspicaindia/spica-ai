"""
Spica AI - SFT (Instruction-Tuning) Training Loop

Fine-tunes an already-pretrained checkpoint on (prompt, response) pairs
with loss masked to the response tokens only (src/data/sft_dataset.py).
Loads model WEIGHTS from the pretrained checkpoint but starts a FRESH
optimizer -- this is a new training phase with its own much-smaller
learning rate, not a continuation of the pretraining run's schedule.

Usage:
    python -m src.training.sft_trainer --init-checkpoint checkpoints/model_10m/latest.pt
    python -m src.training.sft_trainer --init-checkpoint checkpoints/model_10m/latest.pt --resume checkpoints/model_10m_sft/latest.pt
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from src.config.config_loader import load_config
from src.data.sft_dataset import get_sft_dataloader
from src.model.gpt import GPT
from src.tokenizer.qwen_tokenizer import load_tokenizer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import get_lr
from src.utils.logger import get_logger

logger = get_logger("sft_trainer")

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple[GPT, SimpleNamespace]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = SimpleNamespace(**ckpt["model_config"])
    model = GPT(
        vocab_size=model_cfg.vocab_size,
        block_size=model_cfg.block_size,
        n_layer=model_cfg.n_layer,
        n_embd=model_cfg.n_embd,
        n_head=model_cfg.n_head,
        dropout=model_cfg.dropout,
        bias=model_cfg.bias,
        tied_embeddings=model_cfg.tied_embeddings,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model, model_cfg


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
    parser.add_argument("--init-checkpoint", default=None, help="pretrained checkpoint to fine-tune from")
    parser.add_argument("--sft-config", default="configs/sft_config.yaml")
    parser.add_argument("--resume", default=None, help="SFT checkpoint to resume an in-progress SFT run from")
    args = parser.parse_args()

    if not args.init_checkpoint and not args.resume:
        parser.error("need --init-checkpoint (start SFT) or --resume (continue an SFT run)")

    train_cfg = load_config(args.sft_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    init_path = args.resume if args.resume else args.init_checkpoint
    model, model_cfg = build_model_from_checkpoint(init_path, device)
    optimizer = build_optimizer(model, train_cfg.learning_rate, train_cfg.weight_decay)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, map_location=device)
        logger.info(f"resumed SFT from {args.resume} at step {start_step}")
    else:
        logger.info(f"starting SFT fresh from pretrained checkpoint {args.init_checkpoint}")

    tokenizer = load_tokenizer()
    train_loader = get_sft_dataloader(
        ROOT_DIR / train_cfg.train_data, tokenizer, model_cfg.block_size, train_cfg.batch_size, shuffle=True
    )
    val_loader = get_sft_dataloader(
        ROOT_DIR / train_cfg.val_data, tokenizer, model_cfg.block_size, train_cfg.batch_size, shuffle=True
    )
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
    logger.info("SFT training complete, final checkpoint saved")


if __name__ == "__main__":
    main()
