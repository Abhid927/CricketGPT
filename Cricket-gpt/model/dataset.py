import os
import torch
from torch.utils.data import Dataset

class CricketTextDataset(Dataset):
    """
    Loads a whole text file, tokenizes it once, then returns random (x,y) blocks.
    """
    def __init__(self, path: str, tokenizer, block_size: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        ids = tokenizer.encode(text)
        if len(ids) < block_size + 1:
            raise ValueError(f"Not enough tokens in {path} for block_size={block_size}. "
                             f"Got {len(ids)} tokens.")

        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # number of possible start positions
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
