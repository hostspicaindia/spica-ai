"""
Spica AI - Checkpointing

Saves/loads full training state (model weights, optimizer state, step
count) so a run can be resumed exactly where it left off - important on
Colab, where sessions can disconnect at any time.
"""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step: int, model_config) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "model_config": vars(model_config),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)
