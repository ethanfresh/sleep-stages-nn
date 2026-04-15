import os
import glob
import numpy as np
import mne
from scipy.signal import butter, filtfilt

mne.set_log_level("WARNING")

DATA_DIR = "sleep-edf-database-expanded-1.0.0/sleep-cassette"
OUT_DIR  = "data"

STAGE_MAP = {
  "Sleep stage W": 0,
  "Sleep stage 1": 1,
  "Sleep stage 2": 2,
  "Sleep stage 3": 3,
  "Sleep stage 4": 3,
  "Sleep stage R": 4,
}

CHANNELS  = ["EEG Fpz-Cz", "EOG horizontal", "EMG submental"]
SR = 100                 # sampling rate in Hz
EPOCH_LEN = 30 * SR      # 3000 samples per 30-second epoch

def bandpass(signal, lo, hi):
  nyq = SR/2
  b, a = butter(4, [lo/nyq, hi/nyq], btype = "band")
  return filtfilt (b, a, signal)

def load_recording(psg_path, hyp_path):
  raw = mne.io.read_raw_edf(psg_path, preload = True, verbose = False)
  raw = raw.pick_channels(CHANNELS)
  data = raw.get_data()

  data[0] = bandpass(data[0], 0.5, 35)
  data[1] = bandpass(data[1], 0.5, 8)
  data[2] = bandpass(data[2], 10, 49)

  hpy = mne.io.read_raw_edf(hyp_path, preload = True, verbose = False)
  epochs, labels = [], []
  for ann in hpy.annotations:
    if ann["description"] not in STAGE_MAP:
      continue
    onset = int(ann["onset"] * SR)
    end = onset + EPOCH_LEN
    if end > data.shape[1]:
      break
    epochs.append(data[:, onset:end])
    labels.append(STAGE_MAP[ann["description"]])

  return np.array(epochs, dtype = np.float32), np.array(labels, dtype = np.int64)

def main():
  os.makedirs(OUT_DIR, exist_ok = True)
  psg_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.PSG.edf")))

  subjects = {}
  for psg in psg_files:
    subject_id = os.path.basename(psg)[:6]
    subjects.setdefault(subject_id, []).append(psg)

  for subject_id, psgs in subjects.items():
    all_epochs, all_labels = [], []
    for psg_path in psgs:
      hyp_path = psg_path.replace("PSG.edf", "Hypnogram.edf")
      if not os.path.exists(hyp_path):
        continue
      epochs, labels = load_recording(psg_path, hyp_path)
      all_epochs.append(epochs)
      all_labels.append(labels)
      if not all_epochs:
        continue
      X = np.concatenate(all_epochs, axis = 0)
      Y = np.concatenate(all_labels, axis = 0)
      np.save(os.path.join(OUT_DIR, f"{subject_id}_X.npy"), X)
      np.save(os.path.join(OUT_DIR, f"{subject_id}_Y.npy"), Y)
      print(f"{subject_id}: {X.shape[0]} epochs saved")

if __name__ == "__main__":
  main()