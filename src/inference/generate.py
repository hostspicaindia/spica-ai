"""
Spica AI - Text Generation / Inference

Loads a trained checkpoint and generates text from a prompt using
GPT.generate(). The checkpoint's own saved model_config is used to
rebuild the model, so this works for any tier (1M/10M/100M/...)
without needing the config YAML at inference time.

Usage:
    python -m src.inference.generate --prompt "Once upon a time"
    python -m src.inference.generate --prompt "Namaste" --max-new-tokens 100 --top-k 50
    python -m src.inference.generate --checkpoint checkpoints/model_1m/step_1500.pt --prompt "hello"
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.model.gpt import GPT
from src.tokenizer.qwen_tokenizer import load_tokenizer
from src.utils.logger import get_logger

logger = get_logger("generate")

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "checkpoints/model_1m/latest.pt"


def load_model(checkpoint_path: str, device: str) -> GPT:
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
    model.eval()

    logger.info(f"loaded {checkpoint_path} (step {ckpt.get('step', '?')})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    checkpoint_path = ROOT_DIR / args.checkpoint
    model = load_model(str(checkpoint_path), device)
    tokenizer = load_tokenizer()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out[0].tolist())

    print("\n--- generated ---")
    print(text)
    print("-----------------\n")


if __name__ == "__main__":
    main()
