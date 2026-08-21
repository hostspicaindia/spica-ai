"""
Spica AI - Learning Rate Scheduler

Linear warmup (LR rises from ~0 to target over warmup_steps) followed by
cosine decay down to min_lr. Warmup avoids unstable large updates on
freshly-initialized weights; cosine decay lets training settle into a
sharper minimum as it progresses.
"""

import math


def get_lr(step: int, warmup_steps: int, max_steps: int, learning_rate: float, min_lr: float) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)
