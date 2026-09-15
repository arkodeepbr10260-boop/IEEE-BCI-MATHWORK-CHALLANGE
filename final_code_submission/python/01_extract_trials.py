"""
01_extract_trials.py

Extracts labeled EEG trials from the raw IEEE DataPort "7-day Motor Imagery BCI"
dataset (.npz files) and saves them as consolidated NumPy arrays.

INPUT:
    A folder tree of .npz files, each containing:
        - 'signal': (n_samples, n_channels) continuous EEG recording
        - 'mark':   (n_events, 3) event marker array, columns = [time_sec, event_code, unused]
        - 'timestamp': per-sample clock (not used directly here)

OUTPUT:
    X_trials.npy        - (n_trials, 46, 2500) raw epoched EEG, float32
    y_labels.npy         - (n_trials,) integer class labels: 0=Left, 1=Right, 2=Feet, 3=Idle
    meta_subject.npy     - (n_trials,) subject ID string per trial
    meta_session.npy     - (n_trials,) recording date/session ID per trial

METHOD:
    Each trial follows the paradigm: FIXATION -> class cue -> GO_CUE -> 5s MI task window.
    Only the 46-channel recordings are used (the dataset mixes 29- and 46-channel
    acquisitions across subjects; channels are not directly comparable between the two).

NOTE ON RELIABILITY:
    Reading 833 files individually is I/O-bound; this script prints progress and
    checkpoints periodically so a crash does not require restarting from scratch.
"""

import numpy as np
import os
import glob
import time

# --- Configuration ---
DATA_ROOT = r"D:\Desktop\intern\datasets"   # root folder containing all subject/session subfolders
FS = 500                                     # sampling rate (Hz), confirmed from mark timestamps
TRIAL_DURATION_SEC = 5                       # MI task window length, per dataset's published protocol
TARGET_CHANNELS = 46                         # keep only the 46-channel acquisition subset

# Event codes (confirmed against the dataset's official IEEE DataPort description)
LABEL_MAP = {769: 0, 770: 1, 771: 2, 780: 3}  # left hand, right hand, feet, idle
GO_CUE_CODE = 800


def extract_trials_from_file(npz_path):
    """
    Reads one .npz recording and extracts all valid MI trials.

    Returns:
        X: (n_trials_in_file, 46, 2500) float32 array, or None if file doesn't match
           the target channel count or has no valid trials
        y: (n_trials_in_file,) int64 array of labels
    """
    data = np.load(npz_path)
    signal = data["signal"]  # (samples, channels)

    if signal.shape[1] != TARGET_CHANNELS:
        return None, None  # skip 29-channel recordings for channel consistency

    mark = data["mark"]
    codes = mark[:, 1].astype(int)
    times = mark[:, 0]

    trials_X = []
    trials_y = []

    for i, code in enumerate(codes):
        if code not in LABEL_MAP:
            continue

        label = LABEL_MAP[code]

        # The actual task onset is the GO_CUE event immediately following the class cue
        if i + 1 < len(codes) and codes[i + 1] == GO_CUE_CODE:
            onset_time = times[i + 1]
        else:
            onset_time = times[i]  # fallback if GO_CUE marker is missing

        start_sample = int(onset_time * FS)
        end_sample = start_sample + int(TRIAL_DURATION_SEC * FS)

        if end_sample <= signal.shape[0]:
            trial = signal[start_sample:end_sample, :].T  # -> (channels, samples)
            trials_X.append(trial)
            trials_y.append(label)

    if len(trials_y) == 0:
        return None, None

    return np.array(trials_X, dtype=np.float32), np.array(trials_y, dtype=np.int64)


def extract_subject_session(path):
    """Parses subject ID and session date from the file's folder path."""
    parts = path.split(os.sep)
    subject = parts[-3]                        # e.g. 'S9'
    session_folder = parts[-2]                  # e.g. 'S9_20200801'
    session = session_folder.split("_")[-1]     # '20200801'
    return subject, session


def main():
    npz_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npz"), recursive=True)
    print(f"Found {len(npz_files)} .npz files")

    all_X, all_y, all_subjects, all_sessions = [], [], [], []
    start_time = time.time()

    for idx, f in enumerate(npz_files):
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{idx}/{len(npz_files)}] ({elapsed:.1f}s elapsed)", flush=True)

        try:
            X, y = extract_trials_from_file(f)
            if X is not None:
                subj, sess = extract_subject_session(f)
                all_X.append(X)
                all_y.append(y)
                all_subjects.extend([subj] * len(y))
                all_sessions.extend([sess] * len(y))
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.array(all_subjects)
    sessions = np.array(all_sessions)

    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    print(f"Final dataset: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")

    np.save("X_trials.npy", X)
    np.save("y_labels.npy", y)
    np.save("meta_subject.npy", subjects)
    np.save("meta_session.npy", sessions)
    print("Saved X_trials.npy, y_labels.npy, meta_subject.npy, meta_session.npy")


if __name__ == "__main__":
    main()
