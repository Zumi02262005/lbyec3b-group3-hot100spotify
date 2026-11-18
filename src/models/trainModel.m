function R = trainModel(modelName, X, y, cvKfold, opts)
%TRAINMODEL Tune, cross-validate and fit one classification model.
%   R = TRAINMODEL(MODELNAME, X, Y, CVKFOLD, OPTS) runs the same three
%   steps for every model so all results are comparable:
%
%     1. Tune     - Bayesian optimization ('OptimizeHyperparameters') of the
%                   model's hyperparameters, scored by k-fold CV on CVKFOLD.
%     2. Evaluate - k-fold CV with the best hyperparameters on the SAME
%                   CVKFOLD; metrics are reported per fold (mean +/- SD).
%     3. Fit      - final model on the whole training set, with training
%                   and inference time.
%
%   MODELNAME is one of:
%     "logistic"  multinomial logistic regression (baseline), fitcecoc + linear learners
%     "knn"       k-nearest neighbors, fitcknn
%     "tree"      decision tree, fitctree
%     "rf"        random forest, fitcensemble 'Bag'
%     "boost"     boosted trees, fitcensemble 'RUSBoost'
%     "svm"       RBF-kernel SVM, fitcecoc + SVM learners
%     "mlp"       multilayer perceptron, fitcnet
%
%   X is the preprocessed training table (X_train), Y the categorical
%   target (y_train), CVKFOLD the shared stratified k-fold partition.
%
%   OPTS (optional struct):
%     maxEvals     Bayesian optimization evaluations per model (default 30)
%     tuneN        struct of tuning subsample sizes per model, e.g.
%                  struct('svm', 6000). Models not listed tune on all rows.
%     useParallel  run the optimization in parallel (default false)
%     verbose      bayesopt verbosity 0/1/2 (default 1)
%
%   Class imbalance: every model except the MLP uses 'Prior','uniform'.
%   With an empirical prior, fitc* functions renormalize observation
%   weights back to the class frequencies, so inverse-frequency weights
%   alone would have no effect. fitcnet has no 'Prior' option, so it gets
%   inverse-frequency 'Weights' instead. In both cases the CV loss that
%   the optimizer minimizes is the balanced (class-weighted) error.
%
%   rng(42) is reset before tuning, before CV and before the final fit,
%   so each model is reproducible no matter which order models run in.

%% Options
%Fills in defaults for any option that was not given
if nargin < 5
    opts = struct();
end
maxEvals    = getOpt(opts, 'maxEvals', 30);
tuneNAll    = getOpt(opts, 'tuneN', struct());
useParallel = getOpt(opts, 'useParallel', false);
verbose     = getOpt(opts, 'verbose', 1);

modelName = string(modelName);

%% Data
%Uses a numeric matrix because fitcecoc with linear learners does not accept tables
%Predictor names are passed separately so models keep the column names
predNames = X.Properties.VariableNames;
Xm        = X{:, :};

%% Class Order
%Orders the chart_success_tier classes from lowest to highest tier, so confusion
%matrices show whether errors fall into neighboring tiers
cats  = categories(y);
tiers = {'hot_100_entry'; 'top_forty_hit'; 'top_ten_smash'; 'historic_number_one'};
if isempty(setxor(cats, tiers))
    cats = tiers;
end
classNames = categorical(cats, cats);
y          = categorical(y, cats);

%% Model Specification
%Fit function, hyperparameter search space and imbalance handling for this model
spec = modelSpec(modelName, predNames);
hp   = searchSpace(spec, Xm, y);
fit  = @(Xf, yf, p, varargin) fitModel(modelName, Xf, yf, p, classNames, predNames, varargin{:});

fprintf('\n=== %s (%s) ===\n', modelName, spec.family);

%% Tuning Data
%Expensive models can be tuned on a stratified subsample with its own stratified k-fold
%Evaluation and the final fit always use all training rows and cvKfold
rng(42)
N = numel(y);
if isfield(tuneNAll, char(modelName)) && tuneNAll.(modelName) < N
    nSub   = tuneNAll.(modelName);
    cvSub  = cvpartition(y, 'HoldOut', 1 - nSub / N);
    sub    = training(cvSub);
    Xt     = Xm(sub, :);
    yt     = y(sub);
    cvTune = cvpartition(yt, 'KFold', cvKfold.NumTestSets);
    fprintf('Tuning on a stratified subsample of %d rows\n', nnz(sub));
else
    Xt     = Xm;
    yt     = y;
    cvTune = cvKfold;
end

%% Tune
%Bayesian optimization of the hyperparameters, scored by k-fold CV loss
rng(42)
hpo = struct( ...
    'Optimizer', 'bayesopt', ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', maxEvals, ...
    'CVPartition', cvTune, ...
    'ShowPlots', false, ...
    'Verbose', verbose, ...
    'UseParallel', useParallel);

tTune = tic;
[~, opt] = fit(Xt, yt, struct(), ...
    'OptimizeHyperparameters', hp, ...
    'HyperparameterOptimizationOptions', hpo);
tuneTime = toc(tTune);

best = tableToParams(bestPoint(opt));
fprintf('Best CV balanced error during tuning: %.4f (%.1f min)\n', opt.MinObjective, tuneTime / 60);
disp(struct2table(best, 'AsArray', true));

%% Cross-Validate
%Refits the best hyperparameters on every fold of the shared cvKfold partition
rng(42)
cvMdl = fit(Xm, y, best, 'CVPartition', cvKfold);
[cvPred, cvScore] = kfoldPredict(cvMdl);

K = cvKfold.NumTestSets;
foldRows = cell(K, 1);
for k = 1:K
    idx = test(cvKfold, k);
    Mk  = classMetrics(y(idx), cvPred(idx), classNames, cvScore(idx, :));
    foldRows{k} = table(k, Mk.accuracy, Mk.macroF1, Mk.weightedF1, Mk.balancedAcc, Mk.macroAUC, Mk.recall, ...
        'VariableNames', {'Fold', 'Accuracy', 'MacroF1', 'WeightedF1', 'BalancedAcc', 'MacroAUC', 'Recall'});
end
foldMetrics = vertcat(foldRows{:});
pooled      = classMetrics(y, cvPred, classNames, cvScore);

fprintf('CV macro F1: %.4f +/- %.4f | accuracy: %.4f +/- %.4f\n', ...
    mean(foldMetrics.MacroF1), std(foldMetrics.MacroF1), ...
    mean(foldMetrics.Accuracy), std(foldMetrics.Accuracy));

%% Final Fit
%Trains on all training rows with the best hyperparameters and times training
%Inference time is measured on the first 2000 rows and reported per 1000 rows
rng(42)
tFit = tic;
mdl = fit(Xm, y, best);
trainTime = toc(tFit);

nTime        = min(N, 2000);
Xtime        = Xm(1:nTime, :);
inferTime    = timeit(@() predict(mdl, Xtime));
inferMsPer1k = inferTime / nTime * 1000 * 1000;

%% Outputs
%Everything the results table, the paper and the Evaluation Lead need
R.name         = modelName;
R.family       = spec.family;
R.balance      = spec.balance;
R.classNames   = classNames;
R.predNames    = predNames;
R.searchSpace  = describeSpace(hp);
R.tuneRows     = numel(yt);
R.tuneTime     = tuneTime;
R.tuneMinLoss  = opt.MinObjective;
R.tuneTrace    = opt.ObjectiveTrace;
R.bestParams   = best;
R.cvPred       = cvPred;
R.cvScore      = cvScore;
R.foldMetrics  = foldMetrics;
R.pooled       = pooled;
R.model        = mdl;
R.trainTime    = trainTime;
R.inferMsPer1k = inferMsPer1k;
end

%% Helper: modelSpec
%Returns the family, imbalance handling and hyperparameter search ranges of one model
function spec = modelSpec(modelName, predNames)
switch modelName
    case "logistic"
        spec.family  = "Linear";
        spec.fitName = 'fitcecoc';
        spec.learner = 'linear';
        spec.ranges  = struct( ...
            'Lambda', [1e-6, 1], ...
            'Regularization', {{'ridge', 'lasso'}});

    case "knn"
        spec.family  = "Instance-based";
        spec.fitName = 'fitcknn';
        spec.learner = '';
        spec.ranges  = struct( ...
            'NumNeighbors', [1, 5000], ...
            'Distance', {{'euclidean', 'cityblock', 'cosine'}}, ...
            'DistanceWeight', {{'equal', 'inverse', 'squaredinverse'}});

    case "tree"
        spec.family  = "Decision tree";
        spec.fitName = 'fitctree';
        spec.learner = '';
        spec.ranges  = struct( ...
            'MinLeafSize', [1, 2000], ...
            'MaxNumSplits', [1, 5000], ...
            'SplitCriterion', {{'gdi', 'deviance'}});

    case "rf"
        spec.family  = "Ensemble (bagging)";
        spec.fitName = 'fitcensemble';
        spec.learner = 'Tree';
        spec.ranges  = struct( ...
            'NumLearningCycles', [50, 500], ...
            'MinLeafSize', [1, 200], ...
            'NumVariablesToSample', [1, numel(predNames)]);

    case "boost"
        spec.family  = "Ensemble (boosting)";
        spec.fitName = 'fitcensemble';
        spec.learner = 'Tree';
        spec.ranges  = struct( ...
            'NumLearningCycles', [50, 500], ...
            'LearnRate', [0.001, 1], ...
            'MaxNumSplits', [1, 512]);

    case "svm"
        spec.family  = "Kernel";
        spec.fitName = 'fitcecoc';
        spec.learner = 'svm';
        spec.ranges  = struct( ...
            'BoxConstraint', [1e-3, 1e3], ...
            'KernelScale', [1e-2, 1e2]);

    case "mlp"
        spec.family  = "Neural network";
        spec.fitName = 'fitcnet';
        spec.learner = '';
        spec.ranges  = struct( ...
            'NumLayers', [1, 3], ...
            'Layer_1_Size', [10, 200], ...
            'Layer_2_Size', [10, 200], ...
            'Layer_3_Size', [10, 200], ...
            'Activations', {{'relu', 'tanh', 'sigmoid'}}, ...
            'Lambda', [1e-7, 1e-1]);

    otherwise
        error('trainModel:unknownModel', 'Unknown model "%s".', modelName);
end

if modelName == "mlp"
    spec.balance = "inverse-frequency Weights";
else
    spec.balance = "Prior = uniform";
end
end

%% Helper: fitModel
%Fits one model with the hyperparameters in struct p (empty struct = MATLAB defaults)
%The same function is used for tuning, cross-validation and the final fit, so
%extra arguments ('OptimizeHyperparameters', 'CVPartition', ...) go in varargin
%optRes is the bayesopt result when tuning, otherwise empty
function [mdl, optRes] = fitModel(modelName, X, y, p, classNames, predNames, varargin)
common = {'ClassNames', classNames, 'PredictorNames', predNames};
prior  = {'Prior', 'uniform'};
optRes = [];

switch modelName
    case "logistic"
        lin = pick(p, ["Lambda", "Regularization"]);
        t   = templateLinear('Learner', 'logistic', lin{:});
        args = [{'Learners', t, 'Coding', 'onevsall'}, common, prior, varargin];
        if any(strcmp(varargin(1:2:end), 'OptimizeHyperparameters'))
            [mdl, optRes] = fitcecoc(X, y, args{:});
        else
            mdl = fitcecoc(X, y, args{:});
        end

    case "knn"
        hpArgs = pick(p, ["NumNeighbors", "Distance", "DistanceWeight"]);
        mdl = fitcknn(X, y, hpArgs{:}, common{:}, prior{:}, varargin{:});

    case "tree"
        hpArgs = pick(p, ["MinLeafSize", "MaxNumSplits", "SplitCriterion"]);
        mdl = fitctree(X, y, hpArgs{:}, common{:}, prior{:}, varargin{:});

    case "rf"
        treeArgs = pick(p, ["MinLeafSize", "NumVariablesToSample"]);
        ensArgs  = pick(p, "NumLearningCycles");
        t   = templateTree('Reproducible', true, treeArgs{:});
        mdl = fitcensemble(X, y, 'Method', 'Bag', 'Learners', t, ensArgs{:}, ...
                           common{:}, prior{:}, varargin{:});

    case "boost"
        treeArgs = pick(p, "MaxNumSplits");
        ensArgs  = pick(p, ["NumLearningCycles", "LearnRate"]);
        t   = templateTree('Reproducible', true, treeArgs{:});
        mdl = fitcensemble(X, y, 'Method', 'RUSBoost', 'Learners', t, ensArgs{:}, ...
                           common{:}, prior{:}, varargin{:});

    case "svm"
        svmArgs = pick(p, ["BoxConstraint", "KernelScale"]);
        t   = templateSVM('KernelFunction', 'gaussian', 'Standardize', false, ...
                          'CacheSize', 'maximal', svmArgs{:});
        mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsone', ...
                       common{:}, prior{:}, varargin{:});

    case "mlp"
        hpArgs = mlpArgs(p);
        mdl = fitcnet(X, y, 'Weights', classWeights(y), hpArgs{:}, common{:}, varargin{:});
end

if isempty(optRes) && isprop(mdl, 'HyperparameterOptimizationResults')
    optRes = mdl.HyperparameterOptimizationResults;
end
end

%% Helper: searchSpace
%Starts from MATLAB's default search space for the fit function and keeps
%only the hyperparameters listed in spec.ranges, with those ranges
function hp = searchSpace(spec, X, y)
if isempty(spec.learner)
    hp = hyperparameters(spec.fitName, X, y);
else
    hp = hyperparameters(spec.fitName, X, y, spec.learner);
end

names = fieldnames(spec.ranges);
for i = 1:numel(hp)
    hp(i).Optimize = ismember(hp(i).Name, names);
    if hp(i).Optimize
        hp(i).Range = spec.ranges.(hp(i).Name);
    end
end

missingNames = setdiff(names, {hp.Name});
if ~isempty(missingNames)
    error('trainModel:badHyperparameter', 'Not a hyperparameter of %s: %s', ...
        spec.fitName, strjoin(missingNames, ', '));
end
end

%% Helper: describeSpace
%Table of the tuned hyperparameters, their ranges and transforms, for the paper
function S = describeSpace(hp)
hp = hp([hp.Optimize]);
n  = numel(hp);
Name      = strings(n, 1);
Type      = strings(n, 1);
Range     = strings(n, 1);
Transform = strings(n, 1);
for i = 1:n
    Name(i)      = hp(i).Name;
    Type(i)      = hp(i).Type;
    Transform(i) = hp(i).Transform;
    r = hp(i).Range;
    if iscell(r) || isstring(r)
        Range(i) = "{" + strjoin(string(r), ", ") + "}";
    else
        Range(i) = "[" + strjoin(string(r), ", ") + "]";
    end
end
S = table(Name, Type, Range, Transform);
end

%% Helper: tableToParams
%Converts the best point from bayesopt (a one-row table) to a struct
%Categorical values become char so they can be passed as name-value arguments
function p = tableToParams(T)
p = table2struct(T);
f = fieldnames(p);
for i = 1:numel(f)
    v = p.(f{i});
    if iscategorical(v) || isstring(v)
        p.(f{i}) = char(string(v));
    end
end
end

%% Helper: pick
%Returns {Name1, Value1, ...} for the listed fields that exist in p
function nv = pick(p, names)
nv = {};
for n = names
    if isfield(p, n)
        nv = [nv, {char(n), p.(n)}];
    end
end
end

%% Helper: mlpArgs
%Builds fitcnet arguments, turning NumLayers and Layer_i_Size into LayerSizes
function nv = mlpArgs(p)
nv = pick(p, ["Activations", "Lambda"]);
if isfield(p, 'NumLayers')
    sizes = zeros(1, p.NumLayers);
    for i = 1:p.NumLayers
        sizes(i) = p.("Layer_" + i + "_Size");
    end
    nv = [nv, {'LayerSizes', sizes}];
end
end

%% Helper: classWeights
%Inverse-frequency observation weights, scaled so the average weight is 1
function w = classWeights(y)
[g, ~, idx] = unique(y);
counts = accumarray(idx, 1);
w = numel(y) ./ (numel(g) * counts(idx));
end

%% Helper: getOpt
%Returns opts.(name) if it exists, otherwise the default value
function v = getOpt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default;
end
end