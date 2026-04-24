import os
import glob
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import SleepDataset

#config
DATA_DIR   = "data"
BATCH_SIZE = 64
EPOCHS     = 40
LR         = 5e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, default = "cnn", choices = ["cnn", "cnn_lstm"])
parser.add_argument("--channels", nargs = "+", default = ["eeg", "eog", "emg"], choices = ["eeg", "eog", "emg"])
args = parser.parse_args()
MODEL = args.model
CHANNELS = args.channels


# split subjects
all_subjects = sorted(set(
    os.path.basename(f)[:6]
    for f in glob.glob(os.path.join(DATA_DIR, "*_X.npy"))
))
split      = int(len(all_subjects) * 0.8)
train_ids  = all_subjects[:split]
test_ids   = all_subjects[split:]

#dataset and sampler
train_dataset = SleepDataset(DATA_DIR, train_ids, channels = CHANNELS)
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
stage_counts = np.bincount(train_labels, minlength=5).astype(np.float32)
sample_weights = torch.tensor([1.0 / stage_counts[l] for l in train_labels])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(SleepDataset(DATA_DIR, test_ids, channels=CHANNELS), batch_size=BATCH_SIZE)

# model 
if MODEL == "cnn":
  from models.cnn import SleepCNN
  model = SleepCNN(n_channels = len(CHANNELS)).to(DEVICE)
elif MODEL == "cnn_lstm":
  from models.cnn_lstm import SleepCNNLSTM
  model = SleepCNNLSTM(n_channels = len(CHANNELS)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
# training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)

    train_acc  = correct / total
    train_loss = train_loss / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
      for X, y in test_loader:
          X, y = X.to(DEVICE), y.to(DEVICE)
          out  = model(X)
          val_correct += (out.argmax(1) == y).sum().item()
          val_total   += X.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"sleep_{MODEL}_best.pt")
        print(f"  → New best saved ({val_acc:.3f})")

torch.save(model.state_dict(), f"sleep_{MODEL}.pt")
print(f"Model saved to sleep_{MODEL}.pt")