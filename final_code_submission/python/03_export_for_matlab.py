"""
03_export_for_matlab.py

Converts the Python-computed ERD features into a .mat file for MATLAB, which is
the challenge's required modeling platform ("MATLAB or MATLAB with Python").

Labels are shifted from Python's 0-indexed convention (0,1,2,3) to MATLAB's
1-indexed convention (1,2,3,4) to avoid class-index bugs downstream.

INPUT:  erd_features.npy, erd_labels.npy, erd_subjects.npy (from script 02)
OUTPUT: erd_data_for_matlab.mat
"""

import numpy as np
from scipy.io import savemat

features = np.load("erd_features.npy")
labels = np.load("erd_labels.npy")
subjects = np.load("erd_subjects.npy")

labels_matlab = labels + 1  # MATLAB classes are conventionally 1-indexed

savemat("erd_data_for_matlab.mat", {
    'features': features,
    'labels': labels_matlab,
    'subjects': subjects
})

print("Saved erd_data_for_matlab.mat")
print("Features shape:", features.shape)
print("Labels shape:", labels_matlab.shape)
print("Label range:", labels_matlab.min(), "to", labels_matlab.max())
