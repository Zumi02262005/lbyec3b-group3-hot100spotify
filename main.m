%  PREPROCESSING
% ------------------------------------------------------------------------
clear; clc; close all;
rng(100)
data = readtable('billboard100.csv');

[data_uniq, ~] = unique(data, 'stable');

X_uniq = data_uniq{:, [7, 10, 11, 13, 15:20]};

featureNames = data.Properties.VariableNames([7, 10, 11, 13, 15:20]);

cv = cvpartition(size(X_uniq, 1), 'HoldOut', 0.25);

X_train_raw = X_uniq(training(cv), :);
X_test_raw  = X_uniq(test(cv), :);

med_train = median(X_train_raw, 1, 'omitnan');

X_train_imp = X_train_raw;
X_test_imp  = X_test_raw;

for i = 1:size(X_train_raw, 2)

    X_train_imp(isnan(X_train_imp(:,i)),i) = med_train(i);

    X_test_imp(isnan(X_test_imp(:,i)),i) = med_train(i);

end

Q1 = quantile(X_train_imp, 0.25);
Q3 = quantile(X_train_imp, 0.75);

IQR = Q3 - Q1;

lowerBound = Q1 - 1.5 * IQR;
upperBound = Q3 + 1.5 * IQR;

X_train_capped = min(max(X_train_imp, lowerBound), upperBound);
X_test_capped  = min(max(X_test_imp, lowerBound), upperBound);

skew_values = skewness(X_train_capped);

skewed_features = abs(skew_values) > 1;

X_train_transformed = X_train_capped;
X_test_transformed  = X_test_capped;

for i = 1:size(X_train_capped, 2)

    if skewed_features(i)

        X_train_transformed(:,i) = ...
            sign(X_train_capped(:,i)) .* ...
            log1p(abs(X_train_capped(:,i)));

        X_test_transformed(:,i) = ...
            sign(X_test_capped(:,i)) .* ...
            log1p(abs(X_test_capped(:,i)));

    end

end

[X_train, mu, sigma] = zscore(X_train_transformed);

sigma(sigma == 0 | isnan(sigma)) = 1;

X_test = (X_test_transformed - mu) ./ sigma;
