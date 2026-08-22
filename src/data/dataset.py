"""
Spica AI - Token Dataset / DataLoader

Reads the packed binary token files (data/tokenized/{train,val}.bin) and
serves fixed-length (x, y) chunks for next-token-prediction training:
    x = tokens[i : i+block_size]        (input)
    y = tokens[i+1 : i+block_size+1]    (input shifted by one - the target)

Uses numpy memmap so the file is read from disk on demand instead of being
loaded fully into RAM - important once token files get large at bigger
model tiers.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ROOT_DIR = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = ROOT_DIR / "data" / "tokenized"


class TokenDataset(Dataset):
    def __init__(self, split: str, block_size: int):
        path = TOKENIZED_DIR / f"{split}.bin"
        # memmap = lazy, page-in-on-access view of the file; does not load it
        # into RAM up front.
        self.data = np.memmap(path, dtype=np.int32, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        # last valid start index needs block_size+1 tokens ahead of it (x and y)
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def get_dataloader(
    split: str,
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """batch_size is PER PROCESS (per GPU) under DDP -- effective global
    batch size is batch_size * world_size, since each rank pulls its own
    shard via DistributedSampler and runs it through its own model replica.
    """
    dataset = TokenDataset(split, block_size)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
