"""Character-level dataset and tokenizer.

A *character-level* model keeps the project self-contained: the "tokenizer" is
just the sorted set of unique characters in the corpus, so there is nothing to
download or train. Each character maps to an integer id; the model learns to
predict the next character.
"""

from __future__ import annotations

from pathlib import Path

import torch


class CharTokenizer:
    """Maps characters <-> integer ids based on a corpus's character set."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CharDataset:
    """Holds the encoded corpus and serves random training batches."""

    def __init__(self, text: str, block_size: int, split: float = 0.9) -> None:
        self.tokenizer = CharTokenizer(text)
        self.block_size = block_size
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(len(data) * split)
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, device="cpu", generator=None):
        """Return ``(x, y)`` where y is x shifted one character to the right.

        Predicting the next character at every position is what makes this a
        language model: input ``"hell"`` -> targets ``"ello"``.
        """
        data = self.train_data if split == "train" else self.val_data
        # Random start indices for each sequence in the batch.
        ix = torch.randint(len(data) - self.block_size, (batch_size,),
                           generator=generator)
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
