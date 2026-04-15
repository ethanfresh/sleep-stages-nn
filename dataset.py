import os
import glob
import numpy as np
from torch.utils.data import Dataset

class SleepDataset(Dataset):
  def __init__(self, data_dir, subject_ids=None):
    self.samples = []

    pattern = os.path.join(data_dir, "*_X.npy")
    x_files = sorted(glob.glob(pattern))

    for x_path in x_files:
      subject_id = os.path.basename(x_path)[:6]
      if subject_ids is not None and subject_id not in subject_ids:
        continue
      y_path = x_path.replace("_X.npy", "_Y.npy")
      X = np.load(x_path)
      y = np.load(y_path)
      for epoch, label in zip(X, y):
        self.samples.append((epoch, label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    epoch, label = self.samples[idx]
    epoch = epoch.astype(np.float32)
    epoch = (epoch - epoch.mean(axis=1, keepdims=True)) / (epoch.std(axis=1, keepdims=True) + 1e-8)
    return epoch, int(label)