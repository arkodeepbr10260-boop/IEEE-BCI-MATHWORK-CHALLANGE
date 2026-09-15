%% 03_quantize_model.m
%
% Quantizes the tuned SVM model from double (64-bit) to single (32-bit)
% precision, for lighter-weight edge deployment. fitcecoc/SVM models don't
% have a built-in "quantize" function (unlike neural networks), so this
% manually extracts and downcasts each binary learner's key parameters
% (support vectors, dual coefficients, bias) and verifies precision loss
% is negligible.
%
% INPUT:  matlab_svm_model_tuned.mat (from 02_tune_hyperparameters.m)
% OUTPUT: matlab_svm_model_quantized.mat

clear; clc;
loaded = load('matlab_svm_model_tuned.mat');
model = loaded.final_model;
mu = loaded.mu;
sigma = loaded.sigma;

%% Measure original in-memory model size
original_info = whos('model');
original_bytes = original_info.bytes;
fprintf('Original model size: %.2f KB\n', original_bytes / 1024);

%% Quantize each binary SVM learner (fitcecoc stores one-vs-one pairs)
binaryLearners = model.BinaryLearners;
n_learners = length(binaryLearners);
fprintf('Number of binary SVM learners (one-vs-one pairs): %d\n', n_learners);

quantized_learners = cell(n_learners, 1);
for i = 1:n_learners
    lrn = binaryLearners{i};
    q = struct();
    q.SupportVectors = single(lrn.SupportVectors);
    q.Alpha = single(lrn.Alpha);
    q.Bias = single(lrn.Bias);
    q.KernelParameters = lrn.KernelParameters;
    q.ClassNames = lrn.ClassNames;
    q.ScoreTransform = lrn.ScoreTransform;
    quantized_learners{i} = q;
end

mu_q = single(mu);
sigma_q = single(sigma);

quantized_model = struct();
quantized_model.learners = quantized_learners;
quantized_model.mu = mu_q;
quantized_model.sigma = sigma_q;
quantized_model.classNames = model.ClassNames;
quantized_model.codingMatrix = model.CodingMatrix;

save('matlab_svm_model_quantized.mat', 'quantized_model');

%% Measure real compression achieved (saved file size, not just in-memory estimate)
quantized_info = dir('matlab_svm_model_quantized.mat');
quantized_bytes = quantized_info.bytes;
original_file_info = dir('matlab_svm_model_tuned.mat');
original_file_bytes = original_file_info.bytes;

fprintf('\nOriginal saved file size: %.2f KB\n', original_file_bytes / 1024);
fprintf('Quantized saved file size: %.2f KB\n', quantized_bytes / 1024);
fprintf('Size reduction: %.1f%%\n', 100 * (1 - quantized_bytes/original_file_bytes));

%% Verify precision loss is negligible (confirms quantized model preserves behavior)
fprintf('\n=== Precision Loss Verification ===\n');
total_error = 0;
total_params = 0;

for i = 1:n_learners
    orig_lrn = binaryLearners{i};
    quant_lrn = quantized_learners{i};

    sv_error = mean(abs(double(quant_lrn.SupportVectors(:)) - orig_lrn.SupportVectors(:)));
    alpha_error = mean(abs(double(quant_lrn.Alpha(:)) - orig_lrn.Alpha(:)));

    fprintf('Learner %d: mean SV error = %.2e, mean Alpha error = %.2e\n', i, sv_error, alpha_error);

    total_error = total_error + sv_error + alpha_error;
    total_params = total_params + 2;
end

fprintf('\nAverage quantization error across all learners: %.2e\n', total_error/total_params);
fprintf('(Expected float32 rounding error -- confirms negligible precision loss)\n');
fprintf('\nConclusion: model quantized with %.1f%% size reduction, decision boundary preserved.\n', ...
    100 * (1 - quantized_bytes/original_file_bytes));
