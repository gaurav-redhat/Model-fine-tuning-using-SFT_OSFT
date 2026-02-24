"""
Dataset utilities for LoRA fine-tuning.

Provides a generic TextClassificationDataset that tokenises text on-the-fly,
plus helper functions to build DataLoaders from HuggingFace datasets or local
CSV/JSON files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizerBase


class TextClassificationDataset(Dataset):
    """Simple dataset that holds (text, label) pairs and tokenises lazily."""

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128,
    ):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # squeeze out the batch dim that return_tensors="pt" adds
        item = {}
        for key, value in encoding.items():
            item[key] = value.squeeze(0)

        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_from_huggingface(
    dataset_name,
    tokenizer,
    text_column="text",
    label_column="label",
    max_length=128,
    split="train",
):
    """Load a HuggingFace dataset and wrap it as a TextClassificationDataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    texts = ds[text_column]
    labels = ds[label_column]

    return TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def load_from_csv(
    path,
    tokenizer,
    text_column="text",
    label_column="label",
    max_length=128,
):
    """Load a local CSV file and wrap it as a TextClassificationDataset."""
    import pandas as pd

    df = pd.read_csv(path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    return TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def build_dataloaders(
    train_dataset,
    val_dataset=None,
    batch_size=16,
    num_workers=2,
):
    """Create train (shuffled) and optional validation DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
