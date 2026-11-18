%% TRAIN MODELS - Billboard Hot 100 & Spotify Audio Features (LBYEC3B Team 3)
%Tunes, cross-validates and fits every candidate model with trainModel
%Run main.m first so data/processed/preprocessed.mat exists
%The test set is NOT loaded here, it is only used once at the end for the selected models
clear; clc; close all;
rng(42)

%% Paths
%Adds src/modeling in case the folder is not on the MATLAB Project path yet
if ~exist('trainModel', 'file')
    addpath(fullfile('src', 'modeling'));
end

%% Load Preprocessed Data
%Training set, target and the shared stratified 5-fold partition from main.m
load(fullfile('data', 'processed', 'preprocessed.mat'), 'X_train', 'y_train', 'cvKfold');
fprintf('Training set: %d rows, %d features, %d folds\n', height(X_train), width(X_train), cvKfold.NumTestSets);

%% Class Order
%Same lowest-to-highest tier order that trainModel uses
cats  = categories(y_train);
tiers = {'hot_100_entry'; 'top_forty_hit'; 'top_ten_smash'; 'historic_number_one'};
if isempty(setxor(cats, tiers))
    cats = tiers;
end
classNames = categorical(cats, cats);
y_train    = categorical(y_train, cats);

%% Settings
%models: which models to run, in order
%maxEvals: Bayesian optimization evaluations per model
%tuneN: tuning subsample size for expensive models (evaluation still uses all rows)
%useParallel: true needs the Parallel Computing Toolbox, but runs are then not exactly reproducible
%resume: skip models whose results file already exists
models = ["logistic", "knn", "tree", "rf", "boost", "svm", "mlp"];

opts.maxEvals    = 30;
opts.tuneN       = struct('svm', 6000);
opts.useParallel = false;
opts.verbose     = 1;
resume           = true;

%% Output Folders
%results/models holds one .mat per model, results/figures holds the plots
outDir = 'results';
modelDir = fullfile(outDir, 'models');
figDir = fullfile(outDir, 'figures');
if ~isfolder(modelDir)
    mkdir(modelDir);
end
if ~isfolder(figDir)
    mkdir(figDir);
end

%% Majority-Class Baseline
%Predicts the most frequent class of each training fold, the score every model must beat
K = cvKfold.NumTestSets;
basePred = y_train;
baseRows = cell(K, 1);
for k = 1:K
    majority = mode(y_train(training(cvKfold, k)));
    idx = test(cvKfold, k);
    basePred(idx) = majority;
    Mk = classMetrics(y_train(idx), basePred(idx), classNames);
    baseRows{k} = table(k, Mk.accuracy, Mk.macroF1, Mk.weightedF1, Mk.balancedAcc, NaN, Mk.recall, ...
        'VariableNames', {'Fold', 'Accuracy', 'MacroF1', 'WeightedF1', 'BalancedAcc', 'MacroAUC', 'Recall'});
end
baseFolds = vertcat(baseRows{:});
fprintf('Majority baseline: accuracy %.4f, macro F1 %.4f\n', mean(baseFolds.Accuracy), mean(baseFolds.MacroF1));

%% Train Models
%Each model is saved as soon as it finishes so a crash does not lose earlier models
results = struct();
for m = models
    file = fullfile(modelDir, m + ".mat");
    if resume && isfile(file)
        fprintf('\nLoading saved results for %s\n', m);
        S = load(file, 'R');
        R = S.R;
    else
        R = trainModel(m, X_train, y_train, cvKfold, opts);
        save(file, 'R', '-v7.3');
    end
    results.(m) = R;
end

%% Results Table
%Mean and SD across the CV folds for every model, plus the majority baseline
rows = cell(numel(models) + 1, 1);
rows{1} = summaryRow("majority", "Baseline", baseFolds, NaN, NaN, NaN, NaN, classNames);
for i = 1:numel(models)
    R = results.(models(i));
    rows{i + 1} = summaryRow(R.name, R.family, R.foldMetrics, R.trainTime, R.inferMsPer1k, ...
                             R.tuneTime / 60, R.tuneRows, classNames);
end
cvResults = sortrows(vertcat(rows{:}), 'MacroF1', 'descend');
disp(cvResults)

writetable(cvResults, fullfile(outDir, 'cv_results.csv'));
save(fullfile(outDir, 'cv_results.mat'), 'cvResults', 'baseFolds');

%% Search Spaces
%Tuned hyperparameters, ranges and best values of every model, for the paper
for m = models
    R = results.(m);
    fprintf('\n%s - tuned on %d rows, balance: %s\n', m, R.tuneRows, R.balance);
    disp(R.searchSpace)
    disp(struct2table(R.bestParams, 'AsArray', true))
end

%% Figure: Macro F1 Comparison
%Mean CV macro F1 with SD error bars, dashed line is the majority baseline
modelRows = cvResults(cvResults.Model ~= "majority", :);
fig1 = figure('Name', 'CV macro F1');
nModels = height(modelRows);
bar(1:nModels, modelRows.MacroF1);
hold on
errorbar(1:nModels, modelRows.MacroF1, modelRows.MacroF1_SD, 'k.', 'LineWidth', 1);
xticks(1:nModels);
xticklabels(modelRows.Model);
yline(mean(baseFolds.MacroF1), '--', 'Majority baseline');
hold off
ylabel('Macro F1 (5-fold CV)');
title('Cross-validated macro F1 by model');
exportgraphics(fig1, fullfile(figDir, 'cv_macroF1.png'), 'Resolution', 200);

%% Figure: Confusion Matrices
%One row-normalized CV confusion matrix per model, so the diagonal is each class's recall
%Each model gets its own figure so the chart is centered and can be captioned separately in the paper
%Underscores in class names are replaced with spaces so they are not drawn as subscripts
plotNames = strrep(cellstr(classNames), '_', ' ');
yPlot = renamecats(y_train, cellstr(classNames), plotNames);
for m = models
    R = results.(m);
    predPlot = renamecats(categorical(R.cvPred, cellstr(classNames)), cellstr(classNames), plotNames);

    figCM = figure('Name', "CV confusion matrix - " + m, 'Position', [100 100 700 600]);
    cc = confusionchart(figCM, yPlot, predPlot, 'Normalization', 'row-normalized');
    sortClasses(cc, plotNames);
    cc.Title = sprintf('%s (CV macro F1 %.3f)', R.name, mean(R.foldMetrics.MacroF1));
    cc.XLabel = 'Predicted tier';
    cc.YLabel = 'True tier';

    exportgraphics(figCM, fullfile(figDir, "cv_confusion_" + m + ".png"), 'Resolution', 200);
end

%% Helper: summaryRow
%One results-table row: mean and SD of the fold metrics, per-class recall and timings
function row = summaryRow(name, family, F, trainTime, inferMs, tuneMin, tuneRows, classNames)
recallMean = mean(F.Recall, 1);
row = table(string(name), string(family), ...
    mean(F.MacroF1), std(F.MacroF1), mean(F.Accuracy), std(F.Accuracy), ...
    mean(F.BalancedAcc), mean(F.WeightedF1), mean(F.MacroAUC), ...
    trainTime, inferMs, tuneMin, tuneRows, ...
    'VariableNames', {'Model', 'Family', 'MacroF1', 'MacroF1_SD', 'Accuracy', 'Accuracy_SD', ...
                      'BalancedAcc', 'WeightedF1', 'MacroAUC', 'TrainTime_s', 'InferMs_per1k', ...
                      'TuneTime_min', 'TuneRows'});
recallNames = matlab.lang.makeValidName("Recall_" + string(classNames));
for c = 1:numel(classNames)
    row.(recallNames(c)) = recallMean(c);
end
end