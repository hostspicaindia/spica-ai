"""
Spica AI - SFT (Instruction-Tuning) Dataset

Unlike the pretraining TokenDataset (flat token stream, next-token
prediction over everything), SFT trains on (prompt, response) pairs with
loss computed ONLY on the response tokens -- the prompt is masked out with
label = -100, PyTorch cross_entropy's default ignore_index, so the model
learns to *generate* answers instead of predicting the prompt text itself.

Expects a JSONL file where each line is {"prompt": str, "response": str}
(see src/data/prepare_sft.py).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, block_size: int):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.examples.append((rec["prompt"], rec["response"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        prompt, response = self.examples[idx]

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response) + [self.tokenizer.eos_token_id]

        # full unshifted sequence, and a same-length label sequence where
        # prompt positions are masked -- loss should only ever be computed
        # on "did the model correctly predict a response/eos token"
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

        # work in a block_size+1 buffer, same convention as the pretraining
        # TokenDataset, so the shift below yields x/y of exactly block_size
        max_len = self.block_size + 1
        if len(input_ids) > max_len:
            # truncate from the LEFT -- keeps the response (end of sequence)
            # intact since that's what loss is computed on; long prompts
            # lose their earliest context instead
            overflow = len(input_ids) - max_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        # standard next-token shift: x[i] predicts the token at labels[i+1]
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def get_sft_dataloader(path, tokenizer, block_size: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = SFTDataset(path, tokenizer, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
