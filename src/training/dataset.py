"""TransactionDataset — wraps the labeled parquet file for PyTorch training.

Each sample is a single transaction description. The dataset tokenizes on the
fly and returns a dict of tensors that the model and loss function expect:

  input_ids      (seq_len,)  token IDs from the tokenizer vocabulary
  attention_mask (seq_len,)  1 for real tokens, 0 for padding
  label          ()          integer class index
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer


class TransactionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label2id: dict[str, int],
        tokenizer: AutoTokenizer,
        max_length: int = 64,
    ):
        self.descriptions = df["description"].tolist()
        self.labels = df["label"].map(label2id).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.descriptions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.descriptions[idx],
            max_length=self.max_length,
            padding="max_length",   # pad every sample to max_length
            truncation=True,        # truncate if somehow longer (rare here)
            return_tensors="pt",    # return PyTorch tensors directly
        )
        return {
            # squeeze removes the batch dim the tokenizer adds: (1, L) -> (L,)
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def stratified_split(
    dataset: TransactionDataset,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test preserving class proportions.

    Uses two successive stratified splits so every class appears in all three
    partitions — critical when some classes have very few examples.
    """
    labels = dataset.labels
    idx = np.arange(len(dataset))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )

    labels_trainval = [labels[i] for i in idx_trainval]
    relative_val = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=relative_val,
        stratify=labels_trainval,
        random_state=seed,
    )

    return Subset(dataset, idx_train), Subset(dataset, idx_val), Subset(dataset, idx_test)
