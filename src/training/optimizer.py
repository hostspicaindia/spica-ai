"""
Spica AI - Optimizer

AdamW with weight decay applied only to weight matrices (Linear, Embedding
weights - anything 2D+), never to biases or LayerNorm parameters (1D).
Decaying those doesn't help and can hurt training stability - standard
GPT-2 practice.
"""

import torch


def build_optimizer(model, learning_rate: float, weight_decay: float, betas=(0.9, 0.95), fused: bool = False):
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # fused=True runs the optimizer.step() update as a single CUDA kernel
    # instead of per-parameter Python-level ops -- free speedup, no accuracy
    # tradeoff. CUDA-only: callers must pass fused=False on CPU (fused=True
    # with CPU params raises, it does not silently fall back).
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)
