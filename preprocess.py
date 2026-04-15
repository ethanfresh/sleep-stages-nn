"""
preprocess.py
-------------
Converts raw Sleep-EDF EDF files into processed NumPy arrays ready for model training.

What this script does, step by step:
  1. Finds all PSG (signal) + Hypnogram (label) file pairs in the sleep-cassette folder
  2. Groups recordings by subject ID so we can do a subject-level train/val/test split
  3. For each recording:
       - Loads the raw EEG, EOG, and EMG signals using MNE
       - Loads the expert sleep stage labels from the hypnogram
       - Cuts the signals into 30-second windows (called "epochs")
       - Applies a bandpass filter to remove noise outside the relevant frequency ranges
       - Normalizes each channel to zero mean and unit variance
  4. Saves everything as .npy files in the data/ folder

Run this once before training:
    python preprocess.py
"""

import os
import glob
import json
import random
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from collections import defaultdict

mne.set_log_level("WARNING")  # suppress MNE's verbose output

# ---------------------------------------------------------------------------
# CONFIGURATION — change these paths if your folder structure is different
# ---------------------------------------------------------------------------

# Where the raw .edf files live
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__),
                            "sleep-edf-database-expanded-1.0.0",
                            "sleep-cassette")

# Where processed .npy files will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Signals to extract (these are the exact channel names inside the EDF files)
CHANNEL_NAMES = {
    "EEG": "EEG Fpz-Cz",
    "EOG": "EOG horizontal",
    "EMG": "EMG submental",
}

# Sleep stage label mapping: annotation string → integer class
STAGE_MAP = {
    "Sleep stage W":  0,   # Wake
    "Sleep stage 1":  1,   # N1 (light sleep)
    "Sleep stage 2":  2,   # N2 (deeper light sleep)
    "Sleep stage 3":  3,   # N3 (deep / slow-wave sleep)
    "Sleep stage 4":  3,   # Stage 4 is merged into N3 (AASM standard)
    "Sleep stage R":  4,   # REM
    # Anything else (movement, unknown) will be discarded
}

EPOCH_DURATION_SEC = 30   # standard clinical epoch length
SAMPLING_RATE      = 100  # Hz — Sleep-EDF is sampled at 100 Hz
SAMPLES_PER_EPOCH  = EPOCH_DURATION_SEC * SAMPLING_RATE  # = 3000 samples

# Train / val / test proportions (by number of subjects, not epochs)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC = 0.15

RANDOM_SEED = 42  # fix seed so the split is reproducible


# ---------------------------------------------------------------------------
# STEP 1: Find all PSG / Hypnogram file pairs
# ---------------------------------------------------------------------------

def find_file_pairs(data_dir):
    """
    Scans the data directory and returns a list of (psg_path, hypnogram_path) tuples.

    Each subject has one PSG file (the raw signal recording) and one Hypnogram file
    (the expert-labeled sleep stages). We match them by their shared prefix —
    e.g., SC4001E0-PSG.edf and SC4001EC-Hypnogram.edf share the prefix 'SC4001'.
    """
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*-PSG.edf")))
    pairs = []

    for psg_path in psg_files:
        basename = os.path.basename(psg_path)           # e.g. SC4001E0-PSG.edf
        prefix   = basename[:6]                          # e.g. SC4001

        # Find the matching hypnogram (same prefix, ends in -Hypnogram.edf)
        hyp_pattern = os.path.join(data_dir, f"{prefix}*-Hypnogram.edf")
        hyp_matches = glob.glob(hyp_pattern)

        if len(hyp_matches) == 1:
            pairs.append((psg_path, hyp_matches[0]))
        else:
            print(f"  [WARNING] Could not find unique hypnogram for {basename}, skipping.")

    print(f"Found {len(pairs)} PSG/Hypnogram pairs.")
    return pairs


# ---------------------------------------------------------------------------
# STEP 2: Extract subject IDs and group recordings by subject
# ---------------------------------------------------------------------------

def get_subject_id(psg_path):
    """
    Extracts the 2-digit subject ID from a filename.

    Filename format: SC4[XX][Y]E0-PSG.edf
      - SC4  = dataset prefix (Sleep Cassette study 4)
      - XX   = subject ID, two digits, e.g. 00, 01, ... 78
      - Y    = night number (1 or 2)

    Examples:
      SC4001E0-PSG.edf  → subject "00", night 1
      SC4012E0-PSG.edf  → subject "01", night 2
      SC4781E0-PSG.edf  → subject "78", night 1
    """
    basename = os.path.basename(psg_path)   # SC4001E0-PSG.edf
    subject_id = basename[3:5]              # characters at index 3 and 4 → "00"
    return subject_id

def group_by_subject(pairs):
    """Returns a dict mapping subject_id → list of (psg_path, hyp_path) tuples."""
    groups = defaultdict(list)
    for psg, hyp in pairs:
        sid = get_subject_id(psg)
        groups[sid].append((psg, hyp))
    print(f"Found {len(groups)} unique subjects.")
    return groups


# ---------------------------------------------------------------------------
# STEP 3: Subject-level train / val / test split
# ---------------------------------------------------------------------------

def split_subjects(groups, train_frac, val_frac, seed):
    """
    Splits subjects (not individual epochs) into train, val, and test sets.

    WHY subject-level? If we split randomly across all epochs, the same person's
    brain patterns would appear in both train AND test. The model would learn to
    recognize that specific person rather than general sleep stage features.
    By splitting at the subject level, every subject in the test set is completely
    unseen during training — simulating real-world deployment.

    Returns three lists of (psg_path, hyp_path) tuples: train, val, test.
    """
    random.seed(seed)
    subject_ids = sorted(groups.keys())
    random.shuffle(subject_ids)

    n_total = len(subject_ids)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    # remaining subjects go to test

    train_ids = subject_ids[:n_train]
    val_ids   = subject_ids[n_train : n_train + n_val]
    test_ids  = subject_ids[n_train + n_val :]

    print(f"Subject split — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # Flatten: each subject may have multiple nights
    train_pairs = [pair for sid in train_ids for pair in groups[sid]]
    val_pairs   = [pair for sid in val_ids   for pair in groups[sid]]
    test_pairs  = [pair for sid in test_ids  for pair in groups[sid]]

    print(f"Recording split — train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs, train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# STEP 4: Signal filtering
# ---------------------------------------------------------------------------

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to remove noise outside [lowcut, highcut] Hz.

    WHY filter? Raw EEG contains power line noise (60 Hz), muscle artifacts,
    and drift. For sleep staging:
      - EEG: keep 0.5–35 Hz (delta, theta, alpha, sigma waves live here)
      - EOG: keep 0.5–8 Hz  (slow eye movements)
      - EMG: keep 10–100 Hz (muscle tone, clipped to Nyquist = fs/2 = 50 Hz here)

    filtfilt applies the filter forward AND backward, giving zero phase shift
    (so the waveform shape is preserved, just cleaned up).
    """
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = min(highcut, nyq - 1) / nyq  # can't exceed Nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# STEP 5: Load one recording and return filtered, normalized epochs + labels
# ---------------------------------------------------------------------------

def process_recording(psg_path, hyp_path, channels_to_use):
    """
    Loads a single PSG + Hypnogram pair and returns:
      - epochs: np.array of shape (n_epochs, n_channels, 3000)
      - labels: np.array of shape (n_epochs,) with integer class labels

    channels_to_use: list of channel keys from CHANNEL_NAMES, e.g. ["EEG"] or
                     ["EEG", "EOG", "EMG"]
    """
    # --- Load raw signals ---
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    channel_labels = [CHANNEL_NAMES[ch] for ch in channels_to_use]

    # Keep only the channels we want
    available = raw.ch_names
    missing   = [c for c in channel_labels if c not in available]
    if missing:
        print(f"  [WARNING] Missing channels {missing} in {os.path.basename(psg_path)}, skipping.")
        return None, None

    raw.pick_channels(channel_labels)
    signals = raw.get_data()   # shape: (n_channels, n_total_samples)

    # --- Apply per-channel bandpass filters ---
    filter_ranges = {
        "EEG Fpz-Cz":      (0.5, 35.0),
        "EOG horizontal":  (0.5,  8.0),
        "EMG submental":   (10.0, 49.0),   # capped below Nyquist (50 Hz)
    }
    for i, ch_label in enumerate(channel_labels):
        low, high = filter_ranges[ch_label]
        signals[i] = bandpass_filter(signals[i], low, high, SAMPLING_RATE)

    # --- Load hypnogram annotations ---
    hyp_raw   = mne.io.read_raw_edf(hyp_path, preload=True, verbose=False)
    annotations = hyp_raw.annotations

    # Build a list of (onset_sample, stage_int) pairs
    epoch_labels = []
    for ann in annotations:
        stage_str = ann["description"]
        if stage_str not in STAGE_MAP:
            continue   # skip unknown/movement epochs
        onset_sample = int(ann["onset"] * SAMPLING_RATE)
        stage_int    = STAGE_MAP[stage_str]
        epoch_labels.append((onset_sample, stage_int))

    # --- Slice signals into 30-second epochs ---
    epochs_list = []
    labels_list = []
    total_samples = signals.shape[1]

    for onset_sample, stage_int in epoch_labels:
        end_sample = onset_sample + SAMPLES_PER_EPOCH
        if end_sample > total_samples:
            break   # don't go past the end of the recording

        epoch = signals[:, onset_sample:end_sample]   # shape: (n_channels, 3000)

        if epoch.shape[1] != SAMPLES_PER_EPOCH:
            continue  # malformed epoch, skip

        epochs_list.append(epoch)
        labels_list.append(stage_int)

    if len(epochs_list) == 0:
        return None, None

    epochs = np.array(epochs_list, dtype=np.float32)   # (n_epochs, n_channels, 3000)
    labels = np.array(labels_list, dtype=np.int64)     # (n_epochs,)

    return epochs, labels


# ---------------------------------------------------------------------------
# STEP 6: Normalize using training set statistics
# ---------------------------------------------------------------------------

def compute_normalization_stats(epochs):
    """
    Computes per-channel mean and standard deviation across ALL training epochs.

    IMPORTANT: we compute these stats ONLY on the training set, then apply the
    same numbers to val and test. If we computed stats on test data, we'd be
    leaking information about the test set into our preprocessing.

    epochs: np.array of shape (N, n_channels, 3000)
    Returns: mean (n_channels,), std (n_channels,)
    """
    # Flatten the time dimension, compute stats per channel
    # epochs shape: (N, C, T) → reshape to (N*T, C) then compute along axis 0
    N, C, T = epochs.shape
    flat = epochs.transpose(1, 0, 2).reshape(C, -1)   # (C, N*T)
    mean = flat.mean(axis=1)   # (C,)
    std  = flat.std(axis=1)    # (C,)
    std[std < 1e-8] = 1e-8     # avoid division by zero
    return mean, std

def normalize(epochs, mean, std):
    """Applies z-score normalization: (x - mean) / std per channel."""
    # epochs: (N, C, T), mean/std: (C,)
    return (epochs - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]


# ---------------------------------------------------------------------------
# STEP 7: Process all recordings for a given split and save
# ---------------------------------------------------------------------------

def process_and_save_split(pairs, split_name, channels_to_use, output_dir,
                           norm_mean=None, norm_std=None):
    """
    Loops through all recordings in a split, processes each one,
    concatenates all epochs, optionally normalizes, and saves as .npy files.

    Returns the epochs and labels arrays (and norm stats if split == "train").
    """
    all_epochs = []
    all_labels = []
    ch_str = "_".join(channels_to_use).lower()   # e.g. "eeg_eog_emg"

    print(f"\nProcessing {split_name} split ({len(pairs)} recordings)...")
    for i, (psg, hyp) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {os.path.basename(psg)}")
        epochs, labels = process_recording(psg, hyp, channels_to_use)
        if epochs is not None:
            all_epochs.append(epochs)
            all_labels.append(labels)

    if len(all_epochs) == 0:
        print(f"  [ERROR] No valid epochs found for {split_name}.")
        return None, None, norm_mean, norm_std

    all_epochs = np.concatenate(all_epochs, axis=0)   # (N_total, C, 3000)
    all_labels = np.concatenate(all_labels, axis=0)   # (N_total,)

    # Compute normalization stats from training data; reuse them for val/test
    if split_name == "train":
        norm_mean, norm_std = compute_normalization_stats(all_epochs)
        print(f"  Normalization — mean: {norm_mean}, std: {norm_std}")

    all_epochs = normalize(all_epochs, norm_mean, norm_std)

    # Print class distribution so we can see the imbalance
    unique, counts = np.unique(all_labels, return_counts=True)
    stage_names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    dist = {stage_names[u]: int(c) for u, c in zip(unique, counts)}
    print(f"  Class distribution: {dist}")
    print(f"  Total epochs: {len(all_epochs)}")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{split_name}_{ch_str}_epochs.npy"), all_epochs)
    np.save(os.path.join(output_dir, f"{split_name}_{ch_str}_labels.npy"), all_labels)
    print(f"  Saved to data/{split_name}_{ch_str}_epochs.npy  and  data/{split_name}_{ch_str}_labels.npy")

    return all_epochs, all_labels, norm_mean, norm_std


# ---------------------------------------------------------------------------
# MAIN — run the full pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Sleep-EDF Preprocessing Pipeline")
    print("=" * 60)

    # Which signal combinations to preprocess.
    # Each entry is a list of keys from CHANNEL_NAMES.
    # We preprocess all three combinations up front so we can run
    # Experiment A (signal ablation) without reprocessing.
    signal_combinations = [
        ["EEG"],                   # Experiment A1: EEG only
        ["EEG", "EOG"],            # Experiment A2: EEG + EOG
        ["EEG", "EOG", "EMG"],     # Experiment A3: EEG + EOG + EMG (full)
    ]

    # Step 1: Find all file pairs
    pairs = find_file_pairs(RAW_DATA_DIR)
    if not pairs:
        print("ERROR: No file pairs found. Check RAW_DATA_DIR path.")
        return

    # Step 2: Group by subject
    groups = group_by_subject(pairs)

    # Step 3: Subject-level split
    train_pairs, val_pairs, test_pairs, train_ids, val_ids, test_ids = \
        split_subjects(groups, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)

    # Step 4: Save the subject split so it can be documented in the paper
    split_record = {
        "train_subject_ids": sorted(train_ids),
        "val_subject_ids":   sorted(val_ids),
        "test_subject_ids":  sorted(test_ids),
        "train_frac": TRAIN_FRAC,
        "val_frac":   VAL_FRAC,
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "subject_split.json"), "w") as f:
        json.dump(split_record, f, indent=2)
    print(f"\nSubject split saved to data/subject_split.json")

    # Step 5: For each signal combination, process all three splits
    for channels_to_use in signal_combinations:
        ch_label = " + ".join(channels_to_use)
        print(f"\n{'=' * 60}")
        print(f"Signal combination: {ch_label}")
        print(f"{'=' * 60}")

        # Train split — also computes normalization stats
        _, _, norm_mean, norm_std = process_and_save_split(
            train_pairs, "train", channels_to_use, OUTPUT_DIR
        )

        # Save the normalization stats so we can apply them at inference time
        ch_str = "_".join(channels_to_use).lower()
        np.save(os.path.join(OUTPUT_DIR, f"norm_mean_{ch_str}.npy"), norm_mean)
        np.save(os.path.join(OUTPUT_DIR, f"norm_std_{ch_str}.npy"),  norm_std)

        # Val and test splits — use train's normalization stats
        process_and_save_split(val_pairs,  "val",  channels_to_use, OUTPUT_DIR, norm_mean, norm_std)
        process_and_save_split(test_pairs, "test", channels_to_use, OUTPUT_DIR, norm_mean, norm_std)

    print(f"\n{'=' * 60}")
    print("Preprocessing complete. All files saved to data/")
    print("Next step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
