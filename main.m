
%  PREPROCESSING
% ------------------------------------------------------------------------
clear; clc; close all;
rng(100) 
data = readtable('billboard100.csv');

[data_uniq, ~] = unique(data, 'stable');

X_uniq = data_uniq{:, [7, 10, 11, 13, 15:20]};

cv = cvpartition(height(data_uniq), 'HoldOut', 0.25);

X_train_raw = X_uniq(training(cv), :);
X_test_raw  = X_uniq(test(cv), :);

med_train = median(X_train_raw, 1, 'omitnan');

X_train_imp = fillmissing(X_train_raw, 'constant', med_train);
X_test_imp  = fillmissing(X_test_raw, 'constant', med_train);

Q1 = quantile(X_train_imp, 0.25);
Q3 = quantile(X_train_imp, 0.75);
IQR = Q3 - Q1;

lowerBound = Q1 - 1.5 * IQR;
upperBound = Q3 + 1.5 * IQR;

X_train_capped = min(max(X_train_imp, lowerBound), upperBound);
X_test_capped  = min(max(X_test_imp, lowerBound), upperBound);

skew_values = skewness(X_train_capped);

skewed_features = skew_values > 1;

X_train_transformed = X_train_capped;
X_test_transformed  = X_test_capped;

valid_features = skewed_features & ...
    all(X_train_capped >= 0, 1);

X_train_transformed(:, valid_features) = ...
    log1p(X_train_capped(:, valid_features));

X_test_transformed(:, valid_features) = ...
    log1p(X_test_capped(:, valid_features));


[X_train, mu, sigma] = zscore(X_train_transformed);

sigma(sigma == 0 | isnan(sigma)) = 1;

X_test = (X_test_transformed - mu) ./ sigma;

