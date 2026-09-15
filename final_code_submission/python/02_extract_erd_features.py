"""
02_extract_erd_features.py

Computes Event-Related Desynchronization (ERD) ratio features for each trial:
    ERD = log( task-window band power / pre-cue rest-window band power )

This is a self-normalizing, per-trial feature that cancels out subject- and
session-level amplitude differences (electrode impedance, skull thickness, etc.),
which raw band power alone does not.

Diagnostic finding that motivated this approach: a paired t-test comparing
rest vs. task band power WITHIN single recordings found 35/46 channels highly
significant (p<0.0001), confirming genuine neural signal is present and that
extraction timing is correct. Raw pooled band power across the FULL dataset
however showed near-chance classification, because between-subject amplitude
variance was swamping the true task-related signal -- ERD normalization
directly targets this.

INPUT:  the raw .npz files (re-read here because rest-window data was not
        saved during trial epoching in 01_extract_trials.py)
OUTPUT: erd_features.npy  - (n_trials, 92) float64  [46 channels x 2 bands]
        erd_labels.npy    - (n_trials,) int64
        erd_subjects.npy  - (n_trials,) str
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
import glob
import os
import time

DATA_ROOT = r"D:\Desktop\intern\datasets"
FS = 500
TRIAL_DURATION_SEC = 5
REST_DURATION_SEC = 1          # pre-cue fixation window used as the rest baseline
LABEL_MAP = {769: 0, 770: 1, 771: 2, 780: 3}
GO_CUE_CODE = 800
TARGET_CHANNELS = 46
BANDS = [(8, 13), (13, 30)]    # mu and beta bands, standard for motor imagery


def bandpass_filter(data, low, high, fs=FS, order=4):
    """4th-order Butterworth band-pass, zero-phase (filtfilt)."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def bandpower_features(trial, fs=FS, bands=BANDS):
    """Welch PSD-based band power per channel per band. trial: (channels, samples)."""
    features = []
    for ch in trial:
        freqs, psd = welch(ch, fs=fs, nperseg=min(256, len(ch)))
        for low, high in bands:
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.trapezoid(psd[idx], freqs[idx]))
    return np.array(features)


def extract_erd_features_from_file(npz_path):
    """Returns (features, labels) with ERD-ratio features per valid trial in this file."""
    data = np.load(npz_path)
    signal = data["signal"]
    if signal.shape[1] != TARGET_CHANNELS:
        return None, None

    mark = data["mark"]
    codes = mark[:, 1].astype(int)
    times = mark[:, 0]

    features_list, labels_list = [], []

    for i, code in enumerate(codes):
        if code not in LABEL_MAP:
            continue
        label = LABEL_MAP[code]
        cue_time = times[i]

        if i + 1 < len(codes) and codes[i + 1] == GO_CUE_CODE:
            go_time = times[i + 1]
        else:
            continue

        rest_start = int((cue_time - REST_DURATION_SEC) * FS)
        rest_end = int(cue_time * FS)
        task_start = int(go_time * FS)
        task_end = task_start + int(TRIAL_DURATION_SEC * FS)

        if rest_start < 0 or task_end > signal.shape[0]:
            continue

        rest_segment = signal[rest_start:rest_end, :].T
        task_segment = signal[task_start:task_end, :].T

        rest_filtered = bandpass_filter(rest_segment.astype(np.float64), 4, 40)
        task_filtered = bandpass_filter(task_segment.astype(np.float64), 4, 40)

        rest_feats = bandpower_features(rest_filtered)
        task_feats = bandpower_features(task_filtered)

        erd_feats = np.log((task_feats + 1e-10) / (rest_feats + 1e-10))

        features_list.append(erd_feats)
        labels_list.append(label)

    if len(labels_list) == 0:
        return None, None

    return np.array(features_list), np.array(labels_list)


def main():
    npz_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npz"), recursive=True)
    print(f"Found {len(npz_files)} files")

    all_features, all_labels, all_subjects = [], [], []
    start_time = time.time()

    for idx, f in enumerate(npz_files):
        if idx % 50 == 0:
            print(f"[{idx}/{len(npz_files)}] ({time.time()-start_time:.1f}s elapsed)", flush=True)

        feats, labels = extract_erd_features_from_file(f)
        if feats is not None:
            all_features.append(feats)
            all_labels.append(labels)
            subject = f.split(os.sep)[-3]
            all_subjects.extend([subject] * len(labels))

        # periodic checkpoint in case of interruption on long runs
        if idx % 100 == 0 and idx > 0 and len(all_features) > 0:
            np.save("erd_features_checkpoint.npy", np.concatenate(all_features, axis=0))
            np.save("erd_labels_checkpoint.npy", np.concatenate(all_labels, axis=0))

    erd_features = np.concatenate(all_features, axis=0)
    erd_labels = np.concatenate(all_labels, axis=0)
    erd_subjects = np.array(all_subjects)

    print(f"\nTotal time: {time.time()-start_time:.1f}s")
    print(f"ERD features shape: {erd_features.shape}")
    print(f"Class distribution: {np.unique(erd_labels, return_counts=True)}")

    np.save("erd_features.npy", erd_features)
    np.save("erd_labels.npy", erd_labels)
    np.save("erd_subjects.npy", erd_subjects)
    print("Saved erd_features.npy, erd_labels.npy, erd_subjects.npy")

    for ckpt in ["erd_features_checkpoint.npy", "erd_labels_checkpoint.npy"]:
        if os.path.exists(ckpt):
            os.remove(ckpt)


if __name__ == "__main__":
    main()
