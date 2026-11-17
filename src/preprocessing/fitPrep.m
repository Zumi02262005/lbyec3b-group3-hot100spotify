function params = fitPrep(T, cfg)
%FITPREP Learn all preprocessing parameters from a TRAINING table only.
%   PARAMS = FITPREP(TTRAIN, CFG). Use APPLYPREP(T, PARAMS, CFG) to
%   transform any table (train, validation fold, or test) with them.
%
%   Order for continuous features:
%   median imputation -> log1p on skewed non-negative features
%   -> IQR capping (only where IQR > 0) -> z-score scaling.

%% Continuous: Imputation
%Fills missing values with the training median of each column
X = T{:, cellstr(cfg.continuous)};

params.med = median(X, 1, 'omitnan');
X = fillmissing(X, 'constant', params.med);

%% Continuous: Log Transform
%Applies log1p to skewed, non-negative features
%Done before capping so capping acts on the less skewed scale
params.logCols = skewness(X) > cfg.skewThresh & all(X >= 0, 1);
X(:, params.logCols) = log1p(X(:, params.logCols));

%% Continuous: Outlier Capping
%IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
%Skipped where IQR = 0 (zero-inflated features), where capping would
%clamp every non-zero value and erase the feature
q    = quantile(X, [0.25 0.75]);
iqrV = q(2, :) - q(1, :);
params.capCols = iqrV > 0;
params.lo = q(1, :) - 1.5 * iqrV;
params.hi = q(2, :) + 1.5 * iqrV;
params.lo(~params.capCols) = -Inf;
params.hi(~params.capCols) =  Inf;
X = min(max(X, params.lo), params.hi);

%% Continuous: Scaling
%Stores normalize's center (C) and scale (S) for z-score scaling
%Zero or NaN scales are set to 1 to avoid division by zero
[~, params.C, params.S] = normalize(X);
params.S(params.S == 0 | isnan(params.S)) = 1;

%% Binary
%Stores the training mode of each binary column for filling missing values
if ~isempty(cfg.binary)
    params.binMode = mode(T{:, cellstr(cfg.binary)}, 1);
else
    params.binMode = [];
end

%% Categorical
%Keeps the cfg.maxLevels most frequent levels of each categorical column
params.levels = cell(1, numel(cfg.categorical));
for i = 1:numel(cfg.categorical)
    s = toLevelString(T.(cfg.categorical(i)));
    s = s(~ismissing(s) & s ~= "");
    [g, ~, idx] = unique(s);
    counts = accumarray(idx, 1);
    [~, order] = sort(counts, 'descend');
    params.levels{i} = g(order(1:min(cfg.maxLevels, numel(order))));
end
end

%% Helper: toLevelString
%Converts a categorical/text/numeric column to strings, keeping NaN as missing
function s = toLevelString(x)
s = string(x);
if isnumeric(x)
    s(isnan(x)) = missing;
end
end