import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from dataset import SleepDataset

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
DATA_DIR    = "data"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model",    type=str, required=True, choices=["cnn", "cnn_lstm"])
parser.add_argument("--channels", nargs="+", default=["eeg", "eog", "emg"], choices=["eeg", "eog", "emg"])
args = parser.parse_args()

if args.model == "cnn":
  from models.cnn import SleepCNN
  model = SleepCNN(n_channels=len(args.channels))
elif args.model == "cnn_lstm":
  from models.cnn_lstm import SleepCNNLSTM
  model = SleepCNNLSTM(n_channels=len(args.channels))

checkpoint = f"sleep_{args.model}_best.pt"
model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
model.to(DEVICE)
model.eval()

all_subjects = sorted(set(
  os.path.basename(f)[:6]
  for f in glob.glob(os.path.join(DATA_DIR, "*_X.npy"))
))
test_ids = all_subjects[int(len(all_subjects) * 0.8):]
test_loader = DataLoader(SleepDataset(DATA_DIR, test_ids, channels=args.channels), batch_size=64)

all_preds, all_labels = [], []
with torch.no_grad():
  for X, y in test_loader:
    X = X.to(DEVICE)
    out = model(X)
    all_preds.extend(out.argmax(1).cpu().tolist())
    all_labels.extend(y.tolist())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"\nModel: {args.model} | Channels: {args.channels}")
print(f"Overall accuracy: {(all_preds == all_labels).mean():.3f}\n")
for i, name in enumerate(STAGE_NAMES):
  mask = all_labels == i
  if mask.sum() == 0:
    continue
  acc = (all_preds[mask] == all_labels[mask]).mean()
  print(f"  {name:5s}: {acc:.3f}  ({mask.sum()} epochs)")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=STAGE_NAMES)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=False)
ax.set_title(f"{args.model.upper()} — {'+'.join(args.channels).upper()}")
plt.tight_layout()

out_path = f"results/{args.model}_{'_'.join(args.channels)}_cm.png"
os.makedirs("results", exist_ok=True)
plt.savefig(out_path)
print(f"\nConfusion matrix saved to {out_path}")