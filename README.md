# Sleep Stage Classifier — DS 340 Final Project
**Team:** Alexander Mannion & Ethan Freshman

Classifies human sleep stages (Wake, N1, N2, N3, REM) from physiological signals
(EEG, EOG, EMG) using a 1D CNN baseline and a CNN+LSTM sequence model.

---

## Project Structure

```
Final Project/
├── sleep-edf-database-expanded-1.0.0/   ← raw data (do not modify)
│   └── sleep-cassette/
├── data/                                 ← processed .npy files (created by preprocess.py)
├── models/
│   ├── cnn.py                            ← 1D CNN baseline
│   └── cnn_lstm.py                       ← CNN + LSTM sequence model
├── results/                              ← saved model checkpoints and figures
├── preprocess.py                         ← Step 1: convert raw EDF → .npy arrays
├── dataset.py                            ← PyTorch Dataset wrapper
├── train.py                              ← Step 2: train a model
├── evaluate.py                           ← Step 3: generate metrics and plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Preprocess the raw data (run once)
```bash
python preprocess.py
```
This reads all 153 EDF recordings, filters and normalizes the signals, segments them
into 30-second epochs, and saves `.npy` files to `data/`. Takes ~10–20 minutes.

### Step 2 — Train a model
```bash
# Train baseline CNN on all three signals
python train.py --model cnn --signals eeg_eog_emg

# Train CNN on EEG only (Experiment A1)
python train.py --model cnn --signals eeg

# Train CNN+LSTM (Experiment B)
python train.py --model cnn_lstm --signals eeg_eog_emg
```

### Step 3 — Evaluate and generate figures
```bash
python evaluate.py --checkpoint results/best_cnn_eeg_eog_emg.pt --signals eeg_eog_emg
```

---

## Experiments

| Exp | Architecture | Signals       | Purpose                            |
|-----|-------------|---------------|------------------------------------|
| A1  | CNN         | EEG only      | Signal ablation baseline           |
| A2  | CNN         | EEG + EOG     | Does eye movement data help?       |
| A3  | CNN         | EEG+EOG+EMG   | Does muscle data help further?     |
| B   | CNN+LSTM    | Best from A   | Does temporal context help?        |

---

## Attribution
- EDF loading: MNE-Python library
- Dataset: PhysioNet Sleep-EDF Database Expanded (Goldberger et al., 2000)
- Architecture inspired by DeepSleepNet (Supratak et al., 2017)
- Code scaffolded with assistance from Claude (Anthropic)
