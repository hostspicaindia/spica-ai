"""
Spica AI - Qwen Tokenizer Wrapper

Wraps the Qwen tokenizer (loaded from HuggingFace Hub) so the rest of the
pipeline (dataset packing, training, inference) always goes through one
consistent interface, regardless of which Qwen checkpoint the tokenizer
comes from.

We only ever use the tokenizer here, never the model weights - Spica AI
trains its own GPT-2 style transformer from scratch.
"""

from transformers import AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger("tokenizer")

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


class SpicaTokenizer:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        logger.info(f"loading Qwen tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        # Qwen tokenizer has no dedicated pad token by default - reuse eos for padding.
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.eos_token_id
        )

        logger.info(
            f"vocab_size={self.vocab_size} eos_token_id={self.eos_token_id} "
            f"pad_token_id={self.pad_token_id}"
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> SpicaTokenizer:
    return SpicaTokenizer(model_name)
