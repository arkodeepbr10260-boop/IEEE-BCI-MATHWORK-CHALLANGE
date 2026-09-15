%% 02_tune_hyperparameters.m
%
% Grid search over the SVM's BoxConstraint (C) and KernelScale, using
% 5-fold cross-validation ON THE TRAINING SET ONLY (test set is never
% touched during tuning, to keep the final evaluation honest).
%
% INPUT:  erd_data_for_matlab.mat
% OUTPUT: hyperparameter_tuning_results.csv - full grid search log
%         matlab_svm_model_tuned.mat        - best model retrained on full training set

clear; clc;
data = load('erd_data_for_matlab.mat');
features = data.features;
labels = data.labels;

%% Same train/test split as 01_train_svm.m, for a fair comparison
rng(42);
cv_outer = cvpartition(labels, 'HoldOut', 0.2, 'Stratify', true);
X_train = features(training(cv_outer), :);
y_train = labels(training(cv_outer));
X_test = features(test(cv_outer), :);
y_test = labels(test(cv_outer));

mu = mean(X_train);
sigma = std(X_train);
X_train_scaled = (X_train - mu) ./ sigma;
X_test_scaled = (X_test - mu) ./ sigma;

%% Grid search
C_values = [0.1, 1, 10, 100];
KernelScale_values = [0.5, 1, 2, 5, 10];

results = [];
best_acc = 0;
best_C = 1;
best_scale = 1;

combo_idx = 0;
total_combos = length(C_values) * length(KernelScale_values);
fprintf('Starting grid search (%d combinations)...\n', total_combos);

for C = C_values
    for scale = KernelScale_values
        combo_idx = combo_idx + 1;
        t = templateSVM('KernelFunction','rbf','BoxConstraint',C, ...
                         'KernelScale',scale,'Standardize',false);

        cv_model = fitcecoc(X_train_scaled, y_train, 'Learners', t, 'KFold', 5);
        cv_acc = 1 - kfoldLoss(cv_model);

        fprintf('[%d/%d] C=%.2f, KernelScale=%.2f -> CV Accuracy: %.4f\n', ...
            combo_idx, total_combos, C, scale, cv_acc);

        results = [results; C, scale, cv_acc]; %#ok<AGROW>

        if cv_acc > best_acc
            best_acc = cv_acc;
            best_C = C;
            best_scale = scale;
        end
    end
end

fprintf('\n=== Best hyperparameters ===\n');
fprintf('C = %.2f, KernelScale = %.2f, CV Accuracy = %.4f\n', best_C, best_scale, best_acc);

%% Retrain final model on the full training set with the best hyperparameters
t_best = templateSVM('KernelFunction','rbf','BoxConstraint',best_C, ...
                      'KernelScale',best_scale,'Standardize',false);
final_model = fitcecoc(X_train_scaled, y_train, 'Learners', t_best);

y_pred = predict(final_model, X_test_scaled);
y_pred = double(y_pred(:));
y_test_col = double(y_test(:));
test_acc = sum(y_pred == y_test_col) / length(y_test_col);

fprintf('\nFinal Test Accuracy (tuned model): %.4f\n', test_acc);

%% Save results
tuning_table = array2table(results, 'VariableNames', {'C','KernelScale','CV_Accuracy'});
writetable(tuning_table, 'hyperparameter_tuning_results.csv');

save('matlab_svm_model_tuned.mat', 'final_model', 'mu', 'sigma', 'best_C', 'best_scale', 'test_acc');
fprintf('\nSaved hyperparameter_tuning_results.csv and matlab_svm_model_tuned.mat\n');
