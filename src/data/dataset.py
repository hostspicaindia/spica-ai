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

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

ROOT_DIR = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = ROOT_DIR / "data" / "tokenized"


class RandomOffsetSampler(Sampler):
    """Draws indices uniformly at random, one at a time, instead of
    RandomSampler's default behavior of eagerly materializing a full
    torch.randperm(len(dataset)) array up front. At 500M-tier scale
    (~669M dataset items) that permutation is a ~5.35GB int64 array
    allocated in one shot the instant iter(dataloader) runs -- exactly
    what caused system RAM to spike immediately at training start.
    Samples with replacement (not a true permutation), which is standard
    practice for this kind of large-corpus pretraining loop and negligible
    at this scale, in exchange for O(1) memory instead of O(n).
    """

    def __init__(self, data_source):
        self.n = len(data_source)

    def __iter__(self):
        for _ in range(self.n):
            yield random.randrange(self.n)

    def __len__(self) -> int:
        return self.n


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
        # NOTE: DistributedSampler has the same eager-torch.randperm cost as
        # the default shuffle=True path below -- not fixed here since DDP
        # is currently unused (abandoned for RTX 5090's lack of NVLink, see
        # trainer.py's docstring), but would need the same treatment if
        # revisited at a scale where it matters.
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    if shuffle:
        return DataLoader(dataset, batch_size=batch_size, sampler=RandomOffsetSampler(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
