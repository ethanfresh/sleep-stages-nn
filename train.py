import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import SleepDataset
from models.cnn import SleepCNN

#config
DATA_DIR   = "data"
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# split subjects
all_subjects = sorted(set(
    os.path.basename(f)[:6]
    for f in glob.glob(os.path.join(DATA_DIR, "*_X.npy"))
))
split      = int(len(all_subjects) * 0.8)
train_ids  = all_subjects[:split]
test_ids   = all_subjects[split:]

#dataset and sampler
train_dataset = SleepDataset(DATA_DIR, train_ids)
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
stage_counts = np.bincount(train_labels, minlength=5).astype(np.float32)
sample_weights = torch.tensor([1.0 / stage_counts[l] for l in train_labels])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader  = DataLoader(SleepDataset(DATA_DIR, test_ids), batch_size=BATCH_SIZE)

# model 
model     = SleepCNN().to(DEVICE)
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
        torch.save(model.state_dict(), "sleep_cnn_best.pt")
        print(f"  → New best saved ({val_acc:.3f})")

torch.save(model.state_dict(), "sleep_cnn.pt")
print("Model saved to sleep_cnn.pt")