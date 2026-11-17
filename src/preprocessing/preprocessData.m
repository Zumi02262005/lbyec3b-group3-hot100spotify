function out = preprocessData(dataFile, cfg)
%PREPROCESSDATA Load, clean, split and preprocess the Billboard/Spotify dataset.
%   OUT = PREPROCESSDATA(DATAFILE, CFG) returns a struct with the preprocessed
%   training and test sets, the CV partitions, and the fitted parameters.
%
%   Leakage rule: everything that uses statistics of the data (medians,
%   skewness, IQR bounds, scaling, category levels) is fitted on the
%   TRAINING partition only (fitPrep) and then applied unchanged to the
%   test set (applyPrep). Only row-level cleaning happens before the split.
%
%   Call rng(42) in main.m BEFORE calling this function.

%% Import
%Reads the CSV and forces the continuous columns to double
opts = detectImportOptions(dataFile);
opts = setvartype(opts, cellstr(cfg.continuous), 'double');
T = readtable(dataFile, opts);
fprintf('Loaded %d rows x %d columns\n', height(T), width(T));

%% Standardize Missing Values
%Treats blanks and "NA" text as missing
%Row-level cleaning uses no statistics, so it is safe before the split
T = standardizeMissing(T, {'', 'NA', 'N/A'});

%% Binary Columns
%Converts text values like "True"/"False" to 1/0
for v = cfg.binary
    x = T.(v);
    if ~isnumeric(x) && ~islogical(x)
        s = lower(strtrim(string(x)));
        x = nan(size(s));
        x(ismember(s, ["true", "1", "yes"]))  = 1;
        x(ismember(s, ["false", "0", "no"]))  = 0;
    end
    T.(v) = double(x);
end

%% Data-Entry Errors
%Values <= 0 in columns that must be positive (e.g. tempo, duration) become missing
for v = cfg.positiveOnly
    bad = T.(v) <= 0;
    T.(v)(bad) = NaN;
    if any(bad)
        fprintf('  %s: %d non-positive values set to missing\n', v, nnz(bad));
    end
end

%% Genre Families
%Maps genre lists like "['dance pop', 'pop', 'pop rap']" to one 0/1 column per family
%A song can belong to several families (multi-hot)
%genre_other = has genres but no family matched, genre_unknown = empty or missing list
%The new columns are added to cfg.binary so they pass through as 0/1
if isfield(cfg, 'genreVar') && ~isempty(cfg.genreVar)
    g = lower(string(T.(cfg.genreVar)));
    unknown = ismissing(g) | strtrim(g) == "" | strtrim(g) == "[]";
    g(unknown) = "";

    fams = string(fieldnames(cfg.genreFamilies))';
    anyMatch = false(height(T), 1);
    newCols = strings(1, 0);
    for f = fams
        hit = contains(g, cfg.genreFamilies.(f));
        T.("genre_" + f) = double(hit);
        anyMatch = anyMatch | hit;
        newCols(end+1) = "genre_" + f;
    end
    T.genre_other   = double(~anyMatch & ~unknown);
    T.genre_unknown = double(unknown);
    newCols = [newCols, "genre_other", "genre_unknown"];

    cfg.binary = [cfg.binary, newCols];
    fprintf('  Genre families: %d songs unknown, %d other (no family matched)\n', ...
        nnz(unknown), nnz(~anyMatch & ~unknown));
end

%% Duplicates
%Removes repeat songs identified by cfg.idVars, not by full-row equality
%Keeps the first occurrence
[~, ia] = unique(T(:, cellstr(cfg.idVars)), 'stable');
fprintf('  Removed %d duplicate songs\n', height(T) - numel(ia));
T = T(ia, :);

%% Target
%Uses cfg.makeTarget if given, otherwise the cfg.target column
%Classification targets become categorical
if isfield(cfg, 'makeTarget') && ~isempty(cfg.makeTarget)
    y = cfg.makeTarget(T);
else
    y = T.(cfg.target);
end
if cfg.task == "classification"
    y = categorical(y);
end

%% Missing Target
%Drops rows with a missing target, the target is never imputed
keep = ~ismissing(y);
fprintf('  Dropped %d rows with missing target\n', nnz(~keep));
T = T(keep, :);
y = y(keep);

%% Hold-Out Split
%Stratified by class for classification, so train and test keep the same class balance
if cfg.task == "classification"
    cvTest = cvpartition(y, 'HoldOut', cfg.testFrac);
else
    cvTest = cvpartition(numel(y), 'HoldOut', cfg.testFrac);
end

Ttrain = T(training(cvTest), :);
ytrain = y(training(cvTest));
Ttest  = T(test(cvTest), :);
ytest  = y(test(cvTest));

%% K-Fold Partition
%Stratified k-fold on the training set only, shared by every model
if cfg.task == "classification"
    cvKfold = cvpartition(ytrain, 'KFold', cfg.kFolds);
else
    cvKfold = cvpartition(numel(ytrain), 'KFold', cfg.kFolds);
end

%% Fit and Apply Preprocessing
%Learns all statistics from the training set, then applies them to both sets
params = fitPrep(Ttrain, cfg);

out.Xtrain = applyPrep(Ttrain, params, cfg);
out.ytrain = ytrain;
out.Xtest  = applyPrep(Ttest, params, cfg);
out.ytest  = ytest;

%% Outputs
%cfg includes the generated genre_* columns
%TtrainRaw/TtestRaw are kept for refitting inside CV folds and for error analysis
out.cvTest    = cvTest;
out.cvKfold   = cvKfold;
out.params    = params;
out.cfg       = cfg;
out.TtrainRaw = Ttrain;
out.TtestRaw  = Ttest;

fprintf('Train: %d rows, Test: %d rows, %d features\n', ...
    height(out.Xtrain), height(out.Xtest), width(out.Xtrain));
end