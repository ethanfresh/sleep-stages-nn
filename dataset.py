"""
dataset.py
----------
Defines a PyTorch Dataset class that loads the preprocessed .npy files
and serves individual epochs (and their labels) to the training loop.

Think of this as the "bridge" between the saved files on disk and PyTorch's
DataLoader, which feeds batches of data into the model during training.

Usage:
    from dataset import SleepDataset
    from torch.utils.data import DataLoader

    train_dataset = SleepDataset(split="train", signals="eeg_eog_emg")
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# Default location of preprocessed files — same as OUTPUT_DIR in preprocess.py
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Human-readable names for each class (used in evaluation plots)
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
NUM_CLASSES  = 5


class SleepDataset(Dataset):
    """
    PyTorch Dataset for the Sleep-EDF sleep stage classification task.

    Each item returned is:
      - epoch:  torch.Tensor of shape (n_channels, 3000) — the filtered,
                normalized physiological signal for one 30-second window
      - label:  torch.long scalar — the sleep stage (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

    Args:
        split   : one of "train", "val", "test"
        signals : one of "eeg", "eeg_eog", "eeg_eog_emg" — must match what
                  was preprocessed by preprocess.py
        data_dir: path to the folder containing the .npy files
    """

    def __init__(self, split: str, signals: str = "eeg_eog_emg", data_dir: str = DATA_DIR):
        super().__init__()

        valid_splits  = ("train", "val", "test")
        valid_signals = ("eeg", "eeg_eog", "eeg_eog_emg")

        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got '{split}'")
        if signals not in valid_signals:
            raise ValueError(f"signals must be one of {valid_signals}, got '{signals}'")

        epochs_path = os.path.join(data_dir, f"{split}_{signals}_epochs.npy")
        labels_path = os.path.join(data_dir, f"{split}_{signals}_labels.npy")

        if not os.path.exists(epochs_path):
            raise FileNotFoundError(
                f"Could not find {epochs_path}. "
                f"Have you run preprocess.py yet?"
            )

        # Load from disk into memory as numpy arrays, then convert to tensors.
        # float32 for signals (matches model weights), int64 for labels (required by CrossEntropyLoss).
        self.epochs = torch.from_numpy(np.load(epochs_path)).float()   # (N, C, 3000)
        self.labels = torch.from_numpy(np.load(labels_path)).long()    # (N,)

        self.split   = split
        self.signals = signals
        self.n_channels = self.epochs.shape[1]

    def __len__(self):
        """Returns the total number of 30-second epochs in this split."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns one epoch and its label.
        PyTorch's DataLoader calls this repeatedly to build each batch.
        """
        return self.epochs[idx], self.labels[idx]

    def class_weights(self):
        """
        Computes inverse-frequency weights for each sleep stage class.

        WHY? Sleep recordings are heavily imbalanced — N2 makes up ~50% of
        a typical night, while N1 might be only 5%. If we treat all errors
        equally, the model learns to predict N2 constantly and still gets
        decent accuracy. Weighting by inverse frequency forces the model to
        pay more attention to rare classes like N1.

        These weights are passed to nn.CrossEntropyLoss(weight=...) during training.

        Returns: torch.Tensor of shape (NUM_CLASSES,)
        """
        counts = torch.bincount(self.labels, minlength=NUM_CLASSES).float()
        # Replace any zero counts with 1 to avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        weights = 1.0 / counts
        # Normalize so the weights sum to NUM_CLASSES (keeps the loss scale stable)
        weights = weights / weights.sum() * NUM_CLASSES
        return weights

    def class_distribution(self):
        """Returns a dict of {stage_name: count} for inspection."""
        counts = torch.bincount(self.labels, minlength=NUM_CLASSES)
        return {STAGE_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)}

    def __repr__(self):
        return (
            f"SleepDataset(split='{self.split}', signals='{self.signals}', "
            f"n_epochs={len(self)}, n_channels={self.n_channels})"
        )


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify the dataset loads
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Running dataset sanity check...\n")

    for signals in ("eeg", "eeg_eog", "eeg_eog_emg"):
        print(f"--- Signal combo: {signals} ---")
        try:
            for split in ("train", "val", "test"):
                ds = SleepDataset(split=split, signals=signals)
                print(f"  {split:5s}: {len(ds):6d} epochs  |  channels: {ds.n_channels}")
                if split == "train":
                    print(f"           Class distribution: {ds.class_distribution()}")
                    print(f"           Class weights:      {ds.class_weights().tolist()}")

            # Test that the DataLoader works — grab one batch
            loader = DataLoader(SleepDataset("train", signals), batch_size=64, shuffle=True)
            batch_epochs, batch_labels = next(iter(loader))
            print(f"  Batch shape: epochs={tuple(batch_epochs.shape)}, labels={tuple(batch_labels.shape)}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        print()
