%% 01_train_svm.m
%
% Trains a multiclass RBF-kernel SVM (via fitcecoc, one-vs-one) to classify
% motor imagery EEG trials into 4 classes: Left Hand, Right Hand, Feet, Idle.
%
% INPUT:  erd_data_for_matlab.mat (produced by Python script 03_export_for_matlab.py)
%         Contains: features (n_trials x 92 ERD-ratio features), labels (1-4)
%
% OUTPUT: matlab_svm_model.mat     - trained model + normalization parameters
%         matlab_predictions.csv   - predictions on the held-out test set
%
% REQUIRES: Statistics and Machine Learning Toolbox (fitcecoc, cvpartition)

clear; clc;

%% Load ERD features and labels
data = load('erd_data_for_matlab.mat');
features = data.features;
labels = data.labels;
subjects = data.subjects;

fprintf('Features size: %d x %d\n', size(features,1), size(features,2));
fprintf('Labels size: %d\n', length(labels));
fprintf('Unique labels: %s\n', mat2str(unique(labels)'));

%% Stratified 80/20 train/test split
rng(42); % fixed seed for reproducibility
cv = cvpartition(labels, 'HoldOut', 0.2, 'Stratify', true);

X_train = features(training(cv), :);
y_train = labels(training(cv));
X_test = features(test(cv), :);
y_test = labels(test(cv));

fprintf('\nTrain size: %d, Test size: %d\n', length(y_train), length(y_test));

%% Normalize features (z-score, using TRAIN statistics only to avoid leakage)
mu = mean(X_train);
sigma = std(X_train);
X_train_scaled = (X_train - mu) ./ sigma;
X_test_scaled = (X_test - mu) ./ sigma;

%% Train multiclass SVM (RBF kernel, one-vs-one via error-correcting output codes)
fprintf('\nTraining multiclass SVM...\n');
t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
model = fitcecoc(X_train_scaled, y_train, 'Learners', t);

%% Evaluate on held-out test set
y_pred = predict(model, X_test_scaled);
y_pred = double(y_pred(:));         % ensure column vector, numeric type
y_test_col = double(y_test(:));

acc = sum(y_pred == y_test_col) / length(y_test_col);
fprintf('\nTest Accuracy: %.4f\n', acc);

%% Confusion matrix visualization
figure;
cm = confusionchart(y_test_col, y_pred, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cm.Title = 'MI Classification Confusion Matrix';
% Class order: 1=Left Hand, 2=Right Hand, 3=Feet, 4=Idle

%% Save model and predictions
save('matlab_svm_model.mat', 'model', 'mu', 'sigma');
fprintf('Saved matlab_svm_model.mat\n');

results = table((1:length(y_test_col))', y_test_col, y_pred, ...
    'VariableNames', {'trial_id','true_label','predicted_label'});
writetable(results, 'matlab_predictions.csv');
fprintf('Saved matlab_predictions.csv\n');
