# Sleep Stage Classifier — DS 340 Final Project
**Team:** Alexander Mannion & Ethan Freshman

Classifies human sleep stages (Wake, N1, N2, N3, REM) from physiological signals
(EEG, EOG, EMG) using a 1D CNN baseline and a CNN+LSTM sequence model.

## Project Structure

```
Final Project/
├── sleep-edf-database-expanded-1.0.0/   ← Raw data
│   └── sleep-cassette/
├── data/                                 ← Processed .npy files (created by preprocess.py)
├── models/
│   ├── cnn.py                            ← 1D CNN baseline
│   └── cnn_lstm.py                       ← CNN + LSTM sequence model
├── results/                              ← Saved model checkpoints and figures
├── preprocessing.py                      ← Step 1: Convert raw EDF → .npy arrays
├── dataset.py                            ← PyTorch Dataset wrapper
├── train.py                              ← Step 2: Used for training models
├── evaluate.py                           ← Step 3: Generate metrics and plots
├── results.md                            ← Stored best results from train/val
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

### Step 1 — Preprocess the raw data (run once)
```bash
python3 preprocessing.py
```
This reads all 153 EDF recordings, filters and normalizes the signals, segments them
into 30-second epochs, and saves `.npy` files to `data/`.

### Step 2 — Train a model
```bash
# Train baseline CNN on all three signals (Experiment A3)
python3 train.py --model cnn --channels eeg eog emg

# Train CNN on EEG only (Experiment A1)
python3 train.py --model cnn --channels eeg

# Train CNN on EEG + EOG (Experiment A2)
python3 train.py --model cnn --channels eeg eog

# Train CNN+LSTM (Experiment B)
python3 train.py --model cnn_lstm --channels eeg eog emg
```

### Step 3 — Evaluate and generate figures
```bash
python3 evaluate.py --model cnn --channels eeg eog emg
python3 evaluate.py --model cnn_lstm --channels eeg eog emg
```

## Experiments

| Exp | Architecture | Signals       | Purpose                            |
|-----|-------------|---------------|------------------------------------|
| A1  | CNN         | EEG only      | Signal ablation baseline           |
| A2  | CNN         | EEG + EOG     | Does eye movement data help?       |
| A3  | CNN         | EEG+EOG+EMG   | Does muscle data help further?     |
| B   | CNN+LSTM    | EEG+EOG+EMG   | Does temporal context help?        |


## Attribution
- EDF loading: MNE-Python library
- Dataset: PhysioNet Sleep-EDF Database Expanded (Goldberger et al., 2000)
- Architecture inspired by DeepSleepNet (Supratak et al., 2017)
- Code scaffolded with assistance from Claude (Anthropic)
