"""
Spica AI - Training Loop

Ties everything together: data, model, optimizer, LR schedule, gradient
clipping, periodic validation, checkpointing.

Single GPU:
    python -m src.training.trainer
    python -m src.training.trainer --resume checkpoints/model_1m/latest.pt

Multi-GPU (DDP, single machine) -- launch with torchrun instead of python,
everything else is identical:
    torchrun --standalone --nproc_per_node=2 -m src.training.trainer --model-config configs/model_100m.yaml --train-config configs/train_config_100m.yaml

batch_size in the train config is PER GPU under DDP -- effective global
batch size is batch_size * nproc_per_node. Not yet tested on real
multi-GPU hardware -- verify with a short run before trusting it for a
long paid one, same as any other config/tier change.

Add --amp for bf16 mixed precision (~1.5-2.5x faster on Tensor Core GPUs,
RTX5090 included) -- resuming a checkpoint saved WITHOUT --amp into a run
WITH it (or vice versa) is safe, since checkpoints always store fp32
master weights/optimizer state regardless; --amp only changes the forward
pass's compute dtype, not what's saved.

Add --compile for torch.compile (JIT-fuses ops into faster GPU kernels,
typically another 20-50% on top of --amp). First call after compiling is
slow (one-time trace+compile, tens of seconds) -- normal, not a hang.
Checkpointing always targets the ORIGINAL uncompiled model reference, so
compiled and uncompiled runs can freely resume from each other's
checkpoints, same as --amp.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.config_loader import load_config
from src.data.dataset import get_dataloader
from src.model.gpt import GPT
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import get_lr
from src.utils.logger import get_logger

logger = get_logger("trainer")

ROOT_DIR = Path(__file__).resolve().parents[2]


def setup_distributed():
    """torchrun sets RANK/LOCAL_RANK/WORLD_SIZE -- their absence means a
    plain single-process `python -m ...` launch, so fall back to the
    original single-GPU/CPU behavior untouched."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size, f"cuda:{local_rank}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return False, 0, 0, 1, device


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
def estimate_val_loss(model, val_loader, eval_iters: int, device: str, use_amp: bool = False) -> float:
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_1m.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume training from")
    parser.add_argument(
        "--amp", action="store_true",
        help="bf16 mixed precision on Tensor Cores (~1.5-2.5x faster on RTX5090/A100/H100). "
             "bf16 needs no loss-scaling (unlike fp16 -- same exponent range as fp32), but this is "
             "the first run using it on this codebase: verify a short run's loss curve looks sane "
             "before trusting it for a long one.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile the model for faster training (~20-50% on top of --amp). First step is "
             "slow (one-time compile) -- expected, not a hang. Checkpoints always save/load the "
             "original uncompiled model, so this is safe to toggle across resumes.",
    )
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)

    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0
    use_amp = args.amp and device.startswith("cuda")
    if device.startswith("cuda"):
        # TF32 speeds up any fp32 matmul that falls outside the autocast
        # region (or all of them, if --amp is off) -- safe and free on
        # Ampere+ (RTX5090 included), negligible precision cost, no
        # loss-scaling or code-shape changes needed unlike fp16.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if is_main:
        logger.info(f"device: {device}  distributed: {is_distributed}  world_size: {world_size}  amp(bf16): {use_amp}")

    # build on the raw (unwrapped) model first -- checkpoint save/load always
    # targets this reference, never the DDP wrapper, so state_dict keys stay
    # identical to the single-GPU case (no "module." prefix) and every other
    # script (generate.py, sft_trainer.py) keeps loading checkpoints unchanged
    raw_model = build_model(model_cfg).to(device)
    # compile wraps raw_model's forward for speed but shares the same
    # underlying parameter tensors -- checkpoint save/load below always
    # targets raw_model directly (never this wrapper), so state_dict keys
    # stay clean and a compiled run can resume an uncompiled checkpoint
    # (or vice versa) without any "_orig_mod." prefix mismatch.
    compiled_model = torch.compile(raw_model) if args.compile else raw_model
    if args.compile and is_main:
        logger.info("torch.compile enabled -- first step will be slow (one-time trace+compile)")
    model = DDP(compiled_model, device_ids=[local_rank]) if is_distributed else compiled_model
    optimizer = build_optimizer(model, train_cfg.learning_rate, train_cfg.weight_decay, fused=device.startswith("cuda"))

    start_step = 0
    if args.resume:
        # target raw_model, not the DDP wrapper, so state_dict keys match
        # (no "module." prefix) -- optimizer.parameters() are the same
        # underlying tensors either way, so its state loads correctly too
        start_step = load_checkpoint(args.resume, raw_model, optimizer, map_location=device)
        if is_main:
            logger.info(f"resumed from {args.resume} at step {start_step}")

    train_loader = get_dataloader(
        "train", model_cfg.block_size, train_cfg.batch_size, shuffle=True,
        distributed=is_distributed, rank=rank, world_size=world_size,
    )
    val_loader = get_dataloader(
        "val", model_cfg.block_size, train_cfg.batch_size, shuffle=True,
        distributed=is_distributed, rank=rank, world_size=world_size,
    )
    train_iter = iter(train_loader)

    checkpoint_dir = ROOT_DIR / train_cfg.checkpoint_dir
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()

    # lr_max_steps (optional config field, defaults to max_steps) sizes the
    # warmup+decay curve relative to THIS run's own step budget rather than
    # the absolute/global step counter. Matters when --resume continues past
    # an earlier schedule's own max_steps (e.g. extending training onto a
    # newly-scaled-up corpus): without this, get_lr() would see the resumed
    # `step` as already deep into decay from the OLD schedule and jump LR
    # straight toward the OLD min_lr's neighborhood -- or, if max_steps was
    # simply raised, jump LR back up toward peak almost instantly (step is
    # already large relative to the new warmup_steps) -- either way a sudden
    # discontinuity Adam's existing moment estimates aren't tuned for. Using
    # `step - start_step` instead makes the resumed run see its own fresh
    # 0-relative warmup+decay, same shape a brand-new run would get, no jump.
    lr_max_steps = getattr(train_cfg, "lr_max_steps", train_cfg.max_steps)

    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step - start_step, train_cfg.warmup_steps, lr_max_steps, train_cfg.learning_rate, train_cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if is_main and step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            logger.info(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s elapsed")

        if step > 0 and step % train_cfg.eval_interval == 0:
            # under DDP each rank evaluates its own val shard -- rank0's
            # logged number is an approximation (smaller/noisier sample),
            # not a full-val-set average, but good enough to track trend
            val_loss = estimate_val_loss(model, val_loader, train_cfg.eval_iters, device, use_amp)
            if is_main:
                logger.info(f"step {step:5d} | val_loss {val_loss:.4f}")

        if is_main and step > 0 and step % train_cfg.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir / f"step_{step}.pt", raw_model, optimizer, step, model_cfg)
            save_checkpoint(checkpoint_dir / "latest.pt", raw_model, optimizer, step, model_cfg)
            logger.info(f"checkpoint saved at step {step}")

    if is_main:
        save_checkpoint(checkpoint_dir / "latest.pt", raw_model, optimizer, train_cfg.max_steps, model_cfg)
        logger.info("training complete, final checkpoint saved")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
