# EEG Motor Imagery Classification — Code Submission

IEEE DataPort – MathWorks Challenge

## Pipeline overview

```
Raw .npz files
      |
      v
[Python] 01_extract_trials.py        -> X_trials.npy, y_labels.npy, meta_subject.npy, meta_session.npy
      |
      v
[Python] 02_extract_erd_features.py  -> erd_features.npy, erd_labels.npy, erd_subjects.npy
      |
      v
[Python] 03_export_for_matlab.py     -> erd_data_for_matlab.mat
      |
      v
[MATLAB] 01_train_svm.m              -> matlab_svm_model.mat, matlab_predictions.csv
      |
      v
[MATLAB] 02_tune_hyperparameters.m   -> hyperparameter_tuning_results.csv, matlab_svm_model_tuned.mat
      |
      v
[MATLAB] 03_quantize_model.m         -> matlab_svm_model_quantized.mat  (final, edge-ready model)
```

## How to run

1. Edit `DATA_ROOT` at the top of `python/01_extract_trials.py` and
   `python/02_extract_erd_features.py` to point at your local copy of the dataset.
2. Run the three Python scripts in order (`01`, `02`, `03`). Each prints progress
   and saves its outputs to the working directory.
3. Open MATLAB, set the working directory to the same folder as the Python outputs.
4. Run the three MATLAB scripts in order (`01`, `02`, `03`).
5. `matlab_svm_model_quantized.mat` is the final deliverable model;
   `matlab_predictions.csv` (from step 01) contains test-set predictions.

## Requirements

- Python 3.x: numpy, scipy, scikit-learn
- MATLAB with Statistics and Machine Learning Toolbox (for `fitcecoc`, `cvpartition`)

## Notes

- Only 46-channel recordings are used (327 of 833 files use a 29-channel montage
  and are excluded for channel consistency).
- The train/test split (80/20, stratified, `random_state`/`rng` = 42) is identical
  across all scripts for a fair, reproducible comparison.
- See the accompanying Technical Report for full methodology, diagnostic findings,
  and results across all tested approaches (band power, CSP, ERD ratio, EEGNet).
