%% EVALUATE MODELS - Billboard Hot 100 & Spotify Audio Features (LBYEC3B Team 3)
%Evaluation phase, run after main.m and train_models.m
%1. Statistical comparison of the top two models (rf vs svm) on the training set
%2. One-time evaluation of the selected model (rf) on the held-out test set
%3. Error analysis of the selected model on the test set
%4. Feature importance and partial dependence of the selected model
%Needs data/processed/preprocessed.mat and results/models/<model>.mat
clear; clc; close all;
rng(42)

%% Settings
%selected: model evaluated on the test set, runnerUp: second model in the significance test
%alpha: significance level of the 5x2cv paired F test
%nBoot: bootstrap resamples for the test-set confidence intervals
%nPerm: permutations per feature group for permutation importance
%pdRows: training rows sampled for the partial dependence plots, nPdVars: features plotted
%runStatTest, runOOB and runLeakFree switch the slower sections on or off
selected = "rf";
runnerUp = "svm";
alpha    = 0.05;
nBoot    = 1000;
nPerm    = 10;
pdRows   = 2000;
nPdVars  = 3;

runStatTest = true;
runOOB      = true;
runLeakFree = true;

%% Output Folders
%Tables go to results/evaluation, figures to results/figures
evalDir = fullfile('results', 'evaluation');
figDir  = fullfile('results', 'figures');
if ~isfolder(evalDir)
    mkdir(evalDir);
end
if ~isfolder(figDir)
    mkdir(figDir);
end

%% Load Data and Models
%Loads the preprocessed split and the saved results of the two models
%X is passed as a numeric matrix in the training column order, as in trainModel
S = load(fullfile('data', 'processed', 'preprocessed.mat'));
X_train = S.X_train;
y_train = S.y_train;
X_test  = S.X_test;
y_test  = S.y_test;
cvKfold = S.cvKfold;
cfg     = S.cfg;

modelDir = fullfile('results', 'models');
Rsel = loadModel(modelDir, selected);
Rrun = loadModel(modelDir, runnerUp);

classNames = Rsel.classNames(:);
predNames  = cellstr(Rsel.predNames);
predNames  = predNames(:)';
dispNames  = strrep(string(classNames), "_", " ");
K          = numel(classNames);

Xtr   = X_train{:, predNames};
Xte   = X_test{:, predNames};
nTest = numel(y_test);

fprintf('Selected model: %s, runner-up: %s\n', selected, runnerUp);
fprintf('Training rows: %d, test rows: %d, features: %d\n', size(Xtr, 1), nTest, numel(predNames));

%% Cross-Validation Recap
%Fold-by-fold macro F1 of both models on the shared cvKfold partition
%These folds were also used for tuning, so this is descriptive only, the formal test is next
F1sel = Rsel.foldMetrics.MacroF1;
F1run = Rrun.foldMetrics.MacroF1;
recap = table((1:numel(F1sel))', F1sel, F1run, F1sel - F1run, ...
    'VariableNames', {'Fold', char(selected), char(runnerUp), 'Difference'});
disp(recap)
fprintf('%s beats %s on %d of %d folds, mean difference %.4f\n', ...
    selected, runnerUp, nnz(F1sel > F1run), numel(F1sel), mean(F1sel - F1run));

%% Statistical Test 1: 5x2cv Paired F Test on Macro F1 (primary)
%Alpaydin's 5x2cv F test, written out because testckfold only allows a custom loss
%when both models are the same type (rf is an ensemble, svm is ECOC)
%5 repetitions of a stratified 2-fold split of the training set; each model is refit
%with its tuned hyperparameters (fitTuned, same settings as trainModel) on one half
%and scored on the other. d(i,j) is the macro F1 difference in repetition i, fold j
%F = sum(d.^2) / (2 * sum(s2)) follows an F(10, 5) distribution under H0
%The test set is not used here
statTbl = table();
statFolds = table();
if runStatTest
    fprintf('\nRunning 5x2cv paired F test on macro F1 (%s vs %s), 10 refits per model...\n', ...
        selected, runnerUp);
    rng(42)
    parts = cell(5, 1);
    for i = 1:5
        parts{i} = cvpartition(y_train, 'KFold', 2);
    end

    tStat = tic;
    f1A = zeros(5, 2);
    f1B = zeros(5, 2);
    for i = 1:5
        for j = 1:2
            tr = training(parts{i}, j);
            te = test(parts{i}, j);
            rng(42)
            mA = fitTuned(selected, Xtr(tr, :), y_train(tr), Rsel.bestParams, classNames, predNames);
            rng(42)
            mB = fitTuned(runnerUp, Xtr(tr, :), y_train(tr), Rrun.bestParams, classNames, predNames);
            MA = classMetrics(y_train(te), predict(mA, Xtr(te, :)), classNames);
            MB = classMetrics(y_train(te), predict(mB, Xtr(te, :)), classNames);
            f1A(i, j) = MA.macroF1;
            f1B(i, j) = MB.macroF1;
            fprintf('  repetition %d, fold %d: %s %.4f, %s %.4f\n', i, j, selected, f1A(i, j), runnerUp, f1B(i, j));
        end
    end
    statTime = toc(tStat);

    d     = f1A - f1B;
    s2    = sum((d - mean(d, 2)).^2, 2);
    Fstat = sum(d(:).^2) / (2 * sum(s2));
    pF    = 1 - fcdf(Fstat, 10, 5);
    hF    = pF < alpha;

    statTbl = table("5x2cv F (macro F1)", string(selected), string(runnerUp), ...
        mean(f1A(:)), std(f1A(:)), mean(f1B(:)), std(f1B(:)), mean(d(:)), Fstat, pF, hF, alpha, statTime / 60, ...
        'VariableNames', {'Test', 'ModelA', 'ModelB', 'ScoreA', 'ScoreA_SD', 'ScoreB', 'ScoreB_SD', ...
                          'MeanDifference', 'Statistic', 'pValue', 'RejectH0', 'Alpha', 'Time_min'});
    statFolds = table(repelem((1:5)', 2), repmat([1; 2], 5, 1), reshape(f1A', [], 1), reshape(f1B', [], 1), ...
        reshape(d', [], 1), 'VariableNames', {'Repetition', 'Fold', char(selected), char(runnerUp), 'Difference'});

    if hF
        fprintf('Macro F1 difference is significant at alpha = %.2f (F = %.3f, p = %.4f)\n', alpha, Fstat, pF);
    else
        fprintf('Macro F1 difference is NOT significant at alpha = %.2f (F = %.3f, p = %.4f)\n', alpha, Fstat, pF);
    end
    writetable(statFolds, fullfile(evalDir, 'stat_test_folds.csv'));
end

%% Statistical Test 2: testckfold on Balanced Error (supporting)
%MATLAB's built-in 5x2cv F test; with different model types only 'classiferror' is allowed
%With 'Prior','uniform' the observation weights sum to 1/K per class, so the
%classification error it compares is the balanced error, the loss used in tuning
%The saved models are passed as templates, so the tuned hyperparameters are kept
if runStatTest
    fprintf('\nRunning testckfold (5x2F, balanced error), 10 more refits per model...\n');
    rng(42)
    tCk = tic;
    [hCk, pCk, e1, e2] = testckfold(Rsel.model, Rrun.model, Xtr, Xtr, y_train, ...
        'Test', '5x2F', 'Alpha', alpha, 'ClassNames', classNames, 'Prior', 'uniform');
    ckTime = toc(tCk);

    ckRow = table("testckfold 5x2F (balanced error)", string(selected), string(runnerUp), ...
        mean(e1(:)), std(e1(:)), mean(e2(:)), std(e2(:)), mean(e1(:) - e2(:)), NaN, pCk, hCk, alpha, ckTime / 60, ...
        'VariableNames', statTbl.Properties.VariableNames);
    statTbl = [statTbl; ckRow];

    if hCk
        fprintf('Balanced error difference is significant at alpha = %.2f (p = %.4f)\n', alpha, pCk);
    else
        fprintf('Balanced error difference is NOT significant at alpha = %.2f (p = %.4f)\n', alpha, pCk);
    end
    disp(statTbl)
    writetable(statTbl, fullfile(evalDir, 'stat_test.csv'));
end

%% Test-Set Evaluation (run once)
%The held-out test set is used here for the first and only time, for the selected model only
%The model is the final fit on all training rows saved by train_models.m
%Nothing in preprocessing or modeling may be changed after these numbers are seen
mdl = Rsel.model;
tPred = tic;
[testPred, testScore] = predict(mdl, Xte);
predTime = toc(tPred);
Mtest = classMetrics(y_test, testPred, classNames, testScore);

%% Majority-Class Baseline on the Test Set
%Predicts the most frequent training class for every test song
majority = mode(y_train);
basePred = repmat(majority, size(y_test));
Mbase = classMetrics(y_test, basePred, classNames);

%% Bootstrap Confidence Intervals
%Resamples the test rows with replacement and recomputes the metrics
%The 2.5th and 97.5th percentiles give a 95% interval
rng(42)
bootVals = zeros(nBoot, 4);
for b = 1:nBoot
    idx = randi(nTest, nTest, 1);
    Mb  = classMetrics(y_test(idx), testPred(idx), classNames);
    bootVals(b, :) = [Mb.accuracy, Mb.macroF1, Mb.weightedF1, Mb.balancedAcc];
end
ci = prctile(bootVals, [2.5 97.5]);

%% Test Results Table
%CV mean and SD (from train_models.m), test value with 95% CI, and the majority baseline
F = Rsel.foldMetrics;
metricNames = ["Accuracy"; "MacroF1"; "WeightedF1"; "BalancedAcc"; "MacroAUC"];
cvMean  = [mean(F.Accuracy); mean(F.MacroF1); mean(F.WeightedF1); mean(F.BalancedAcc); mean(F.MacroAUC)];
cvSD    = [std(F.Accuracy); std(F.MacroF1); std(F.WeightedF1); std(F.BalancedAcc); std(F.MacroAUC)];
testVal = [Mtest.accuracy; Mtest.macroF1; Mtest.weightedF1; Mtest.balancedAcc; Mtest.macroAUC];
ciLo    = [ci(1, :)'; NaN];
ciHi    = [ci(2, :)'; NaN];
baseVal = [Mbase.accuracy; Mbase.macroF1; Mbase.weightedF1; Mbase.balancedAcc; NaN];

testResults = table(metricNames, cvMean, cvSD, testVal, ciLo, ciHi, baseVal, ...
    'VariableNames', {'Metric', 'CV_Mean', 'CV_SD', 'Test', 'Test_CI_Low', 'Test_CI_High', 'MajorityBaseline'});
fprintf('\nTest-set results for %s (%d songs, prediction time %.2f s):\n', selected, nTest, predTime);
disp(testResults)
writetable(testResults, fullfile(evalDir, 'test_results.csv'));

%% Per-Class Test Metrics
%Support, precision, recall and F1 per tier on the test set, next to the CV recall
support  = sum(Mtest.confusion, 2);
cvRecall = mean(F.Recall, 1)';
perClass = table(dispNames, support, Mtest.precision(:), Mtest.recall(:), Mtest.f1(:), cvRecall, ...
    'VariableNames', {'Tier', 'Support', 'Precision', 'Recall', 'F1', 'CV_Recall'});
disp(perClass)
writetable(perClass, fullfile(evalDir, 'test_per_class.csv'));

%% Figure: Test Confusion Matrix
%Row-normalized, so the diagonal is each tier's recall
fig1 = figure('Name', 'Test confusion matrix', 'Position', [100 100 700 560]);
cc = confusionchart(toDisp(y_test, classNames, dispNames), toDisp(testPred, classNames, dispNames), ...
    'Normalization', 'row-normalized');
cc.XLabel = 'Predicted tier';
cc.YLabel = 'True tier';
cc.Title  = sprintf('%s on the test set (macro F1 %.3f)', selected, Mtest.macroF1);
exportgraphics(fig1, fullfile(figDir, "test_confusion_" + selected + ".png"), 'Resolution', 200);

%% Figure: Test ROC Curves
%One-vs-rest ROC curve and AUC for each tier
fig2 = figure('Name', 'Test ROC curves');
rc = rocmetrics(toDisp(y_test, classNames, dispNames), testScore, categorical(dispNames, dispNames));
plot(rc);
title(sprintf('%s one-vs-rest ROC on the test set (macro AUC %.3f)', selected, Mtest.macroAUC));
exportgraphics(fig2, fullfile(figDir, "test_roc_" + selected + ".png"), 'Resolution', 200);

%% Error Analysis: Most Common Confusions
%Off-diagonal cells of the test confusion matrix, largest first
%TierDistance is how many tiers apart the true and predicted tiers are
C = Mtest.confusion;
rowPct = 100 * C ./ max(sum(C, 2), 1);
[ti, pj] = find(~eye(K));
lin = sub2ind([K K], ti, pj);
confusions = table(dispNames(ti), dispNames(pj), C(lin), rowPct(lin), abs(ti - pj), ...
    'VariableNames', {'TrueTier', 'PredictedTier', 'Count', 'PctOfTrueTier', 'TierDistance'});
confusions = sortrows(confusions, 'Count', 'descend');
disp('Most common test-set confusions:');
disp(confusions(1:min(6, height(confusions)), :))

%% Error Analysis: Ordinal Error Distance
%The tiers are ordinal, so an error one tier away is less severe than a jump across tiers
[~, tIdx] = ismember(string(y_test), string(classNames));
[~, pIdx] = ismember(string(testPred), string(classNames));
dist = abs(tIdx - pIdx);
distCount = accumarray(dist + 1, 1, [K 1]);
distTbl = table((0:K - 1)', distCount, 100 * distCount / nTest, ...
    'VariableNames', {'TierDistance', 'Count', 'Percent'});
disp(distTbl)
nErr = nnz(dist > 0);
fprintf('%.1f%% of songs are within one tier, %.1f%% of errors are to an adjacent tier\n', ...
    100 * mean(dist <= 1), 100 * nnz(dist == 1) / max(nErr, 1));

%% Error Analysis: By Musical Era
%The era is read back from the musical_era_* one-hot columns of X_test
eraCols  = startsWith(predNames, "musical_era_");
eraNames = erase(string(predNames(eraCols)), "musical_era_");
[~, eraIdx] = max(Xte(:, eraCols), [], 2);
era = eraNames(eraIdx);
era = era(:);
byEra = subgroupTable(unique(era), @(g) era == g, y_test, testPred, classNames, "Era");
disp(byEra)
writetable(byEra, fullfile(evalDir, 'test_by_era.csv'));

fig3 = figure('Name', 'Test macro F1 by era');
bar(1:height(byEra), byEra.MacroF1);
hold on
yline(Mtest.macroF1, '--', 'Overall test macro F1');
hold off
xticks(1:height(byEra));
xticklabels(strrep(byEra.Era, "_", " ") + " (n = " + byEra.NumSongs + ")");
xtickangle(20);
xlabel('Musical era');
ylabel('Macro F1 (test set)');
title(sprintf('%s test macro F1 by musical era', selected));
exportgraphics(fig3, fullfile(figDir, "test_by_era_" + selected + ".png"), 'Resolution', 200);

%% Error Analysis: By Genre Family
%Songs can belong to several genre families, so the groups overlap
genreCols  = find(startsWith(predNames, "genre_"));
genreNames = erase(string(predNames(genreCols)), "genre_");
byGenre = subgroupTable(genreNames(:), @(g) Xte(:, genreCols(genreNames == g)) == 1, ...
    y_test, testPred, classNames, "Genre");
disp(byGenre)
writetable(byGenre, fullfile(evalDir, 'test_by_genre.csv'));

%% Feature Importance: Impurity-Based
%predictorImportance sums the split gains of every tree, fast but biased toward
%continuous features with many possible split points
impurity = predictorImportance(mdl);
impurity = impurity(:);

%% Feature Importance: Out-of-Bag Permutation
%Permutes each column in the out-of-bag rows of every tree, training data only
%Needs the full bagged model, so it is skipped (with the reason printed) otherwise
oobImp = NaN(numel(predNames), 1);
if runOOB && ismethod(mdl, 'oobPermutedPredictorImportance')
    fprintf('\nComputing out-of-bag permutation importance...\n');
    oobImp = oobPermutedPredictorImportance(mdl);
    oobImp = oobImp(:);
elseif runOOB
    fprintf('\nSkipping OOB importance: the saved model is a %s, which has no out-of-bag data\n', class(mdl));
end

colImp = table(string(predNames(:)), impurity, oobImp, ...
    'VariableNames', {'Feature', 'Impurity', 'OOBPermuted'});
colImp = sortrows(colImp, 'Impurity', 'descend');
writetable(colImp, fullfile(evalDir, 'feature_importance_columns.csv'));

%% Feature Importance: Grouped Permutation on the Test Set
%One-hot columns of the same categorical feature (key, time_signature, musical_era)
%are permuted together so every row stays a valid one-hot code
%Importance is the drop in test macro F1, averaged over nPerm permutations
groupOf = string(predNames);
for c = string(cfg.categorical)
    groupOf(startsWith(groupOf, c + "_")) = c;
end
groups = unique(groupOf, 'stable');
nG = numel(groups);

fprintf('Computing grouped permutation importance (%d groups x %d permutations)...\n', nG, nPerm);
rng(42)
permDrop = zeros(nG, nPerm);
for g = 1:nG
    cols = groupOf == groups(g);
    for r = 1:nPerm
        Xp = Xte;
        Xp(:, cols) = Xte(randperm(nTest), cols);
        Mp = classMetrics(y_test, predict(mdl, Xp), classNames);
        permDrop(g, r) = Mtest.macroF1 - Mp.macroF1;
    end
end

impSum = zeros(nG, 1);
oobSum = zeros(nG, 1);
for g = 1:nG
    cols = groupOf == groups(g);
    impSum(g) = sum(impurity(cols));
    oobSum(g) = sum(oobImp(cols));
end

groupImp = table(groups(:), sum(groupOf(:) == groups(:)', 1)', mean(permDrop, 2), std(permDrop, 0, 2), ...
    impSum, oobSum, ...
    'VariableNames', {'Feature', 'NumColumns', 'PermMacroF1Drop', 'PermMacroF1Drop_SD', ...
                      'ImpuritySum', 'OOBPermutedSum'});
groupImp = sortrows(groupImp, 'PermMacroF1Drop', 'descend');
disp(groupImp)
writetable(groupImp, fullfile(evalDir, 'feature_importance_groups.csv'));

%% Figure: Permutation Importance
%Mean drop in test macro F1 with SD error bars, most important feature on top
gi = flipud(groupImp);
fig4 = figure('Name', 'Permutation importance', 'Position', [100 100 700 650]);
barh(1:height(gi), gi.PermMacroF1Drop);
hold on
errorbar(gi.PermMacroF1Drop, 1:height(gi), gi.PermMacroF1Drop_SD, 'horizontal', 'k.');
hold off
yticks(1:height(gi));
yticklabels(gi.Feature);
set(gca, 'TickLabelInterpreter', 'none');
xlabel('Drop in test macro F1 when permuted');
title(sprintf('%s grouped permutation importance (%d permutations)', selected, nPerm));
exportgraphics(fig4, fullfile(figDir, "importance_permutation_" + selected + ".png"), 'Resolution', 200);

%% Figure: Impurity Importance
%Top 20 one-hot/feature columns by predictorImportance
nTop = min(20, height(colImp));
ci20 = flipud(colImp(1:nTop, :));
fig5 = figure('Name', 'Impurity importance', 'Position', [100 100 700 600]);
barh(1:nTop, ci20.Impurity);
yticks(1:nTop);
yticklabels(ci20.Feature);
set(gca, 'TickLabelInterpreter', 'none');
xlabel('predictorImportance (impurity)');
title(sprintf('%s top %d columns by impurity importance', selected, nTop));
exportgraphics(fig5, fullfile(figDir, "importance_impurity_" + selected + ".png"), 'Resolution', 200);

%% Partial Dependence
%Predicted probability of each tier as one continuous feature varies, averaged over
%pdRows sampled training songs; the most important continuous features are shown
%x is on the preprocessed scale (log1p where applied, capped, z-scored)
isCont = ismember(groupImp.Feature, string(cfg.continuous));
pdVars = groupImp.Feature(isCont);
pdVars = pdVars(1:min(nPdVars, numel(pdVars)));

rng(42)
pdIdx = randperm(size(Xtr, 1), min(pdRows, size(Xtr, 1)));
Xpd = Xtr(pdIdx, :);

fig6 = figure('Name', 'Partial dependence', 'Position', [100 100 400 * numel(pdVars) 420]);
tl = tiledlayout(fig6, 1, numel(pdVars), 'TileSpacing', 'compact');
for i = 1:numel(pdVars)
    ax = nexttile(tl);
    plotPartialDependence(mdl, char(pdVars(i)), classNames, Xpd, 'ParentAxisHandle', ax);
    title(ax, strrep(pdVars(i), "_", " "));
    xlabel(ax, strrep(pdVars(i), "_", " ") + " (scaled)");
    ylabel(ax, 'Predicted probability');
    legend(ax, cellstr(dispNames), 'Location', 'best');
end
title(tl, sprintf('%s partial dependence (%d training songs)', selected, numel(pdIdx)));
exportgraphics(fig6, fullfile(figDir, "partial_dependence_" + selected + ".png"), 'Resolution', 200);

%% Leak-Free CV Check
%cvKfold in train_models.m indexes X_train, whose preprocessing statistics came from the
%whole training set. Here fitPrep is refit inside each fold from the raw training table,
%and the selected model is refit with its tuned hyperparameters
%Needs TtrainRaw in preprocessed.mat (add it to the save line in main.m to enable)
leakFree = table();
if runLeakFree && isfield(S, 'TtrainRaw')
    fprintf('\nRunning leak-free %d-fold CV for %s...\n', cvKfold.NumTestSets, selected);
    TtrainRaw = S.TtrainRaw;
    nFold = cvKfold.NumTestSets;
    lfF1  = zeros(nFold, 1);
    lfAcc = zeros(nFold, 1);
    lfBal = zeros(nFold, 1);
    for k = 1:nFold
        tr = training(cvKfold, k);
        te = test(cvKfold, k);
        pk = fitPrep(TtrainRaw(tr, :), cfg);
        Xa = alignCols(applyPrep(TtrainRaw(tr, :), pk, cfg), predNames);
        Xb = alignCols(applyPrep(TtrainRaw(te, :), pk, cfg), predNames);
        rng(42)
        mk = fitTuned(selected, Xa, y_train(tr), Rsel.bestParams, classNames, predNames);
        Mk = classMetrics(y_train(te), predict(mk, Xb), classNames);
        lfF1(k)  = Mk.macroF1;
        lfAcc(k) = Mk.accuracy;
        lfBal(k) = Mk.balancedAcc;
    end
    leakFree = table(["standard cvKfold"; "leak-free cvKfold"], ...
        [mean(F.MacroF1); mean(lfF1)], [std(F.MacroF1); std(lfF1)], ...
        [mean(F.Accuracy); mean(lfAcc)], [mean(F.BalancedAcc); mean(lfBal)], ...
        'VariableNames', {'CV', 'MacroF1', 'MacroF1_SD', 'Accuracy', 'BalancedAcc'});
    disp(leakFree)
    writetable(leakFree, fullfile(evalDir, 'leak_free_cv.csv'));
elseif runLeakFree
    fprintf('\nSkipping leak-free CV: TtrainRaw is not in preprocessed.mat\n');
end

%% Save Everything
%One .mat with every evaluation result, for the report and slides
save(fullfile(evalDir, 'evaluation.mat'), 'selected', 'runnerUp', 'statTbl', 'statFolds', 'testResults', ...
    'perClass', 'Mtest', 'Mbase', 'bootVals', 'testPred', 'testScore', 'confusions', 'distTbl', ...
    'byEra', 'byGenre', 'colImp', 'groupImp', 'permDrop', 'leakFree');

%% Summary
%Short printout of the numbers the report needs
fprintf('\n===== Evaluation summary =====\n');
if ~isempty(statTbl)
    fprintf('5x2cv F test %s vs %s: macro F1 %.4f vs %.4f, F = %.3f, p = %.4f\n', selected, runnerUp, ...
        statTbl.ScoreA(1), statTbl.ScoreB(1), statTbl.Statistic(1), statTbl.pValue(1));
    if height(statTbl) > 1
        fprintf('testckfold balanced error %s vs %s: %.4f vs %.4f, p = %.4f\n', selected, runnerUp, ...
            statTbl.ScoreA(2), statTbl.ScoreB(2), statTbl.pValue(2));
    end
end
fprintf('Test macro F1 %.4f (95%% CI %.4f-%.4f) vs majority baseline %.4f\n', ...
    Mtest.macroF1, ci(1, 2), ci(2, 2), Mbase.macroF1);
fprintf('Test accuracy %.4f vs majority baseline %.4f, balanced accuracy %.4f, macro AUC %.4f\n', ...
    Mtest.accuracy, Mbase.accuracy, Mtest.balancedAcc, Mtest.macroAUC);
fprintf('Most important feature (permutation): %s\n', groupImp.Feature(1));

%% Helper: loadModel
%Loads the R struct that train_models.m saved for one model
function R = loadModel(modelDir, name)
file = fullfile(modelDir, name + ".mat");
if ~isfile(file)
    error('evaluate_models:missingModel', 'Run train_models.m first, %s not found.', file);
end
S = load(file, 'R');
R = S.R;
end

%% Helper: toDisp
%Relabels class names with spaces instead of underscores (no TeX subscripts in plots)
%and keeps the low-to-high tier order
function yd = toDisp(y, classNames, dispNames)
yd = categorical(string(y), string(classNames), cellstr(dispNames));
end

%% Helper: subgroupTable
%Test metrics for each subgroup (era or genre); maskFun(g) returns the rows of subgroup g
function T = subgroupTable(groupNames, maskFun, yTrue, yPred, classNames, label)
n   = zeros(numel(groupNames), 1);
acc = NaN(numel(groupNames), 1);
mf1 = NaN(numel(groupNames), 1);
bal = NaN(numel(groupNames), 1);
for i = 1:numel(groupNames)
    rows = maskFun(groupNames(i));
    n(i) = nnz(rows);
    if n(i) > 0
        M = classMetrics(yTrue(rows), yPred(rows), classNames);
        acc(i) = M.accuracy;
        mf1(i) = M.macroF1;
        bal(i) = M.balancedAcc;
    end
end
T = table(string(groupNames(:)), n, acc, mf1, bal, ...
    'VariableNames', {char(label), 'NumSongs', 'Accuracy', 'MacroF1', 'BalancedAcc'});
end

%% Helper: alignCols
%Returns the columns of a fold's preprocessed table in the training column order
%A one-hot level that is not in the fold's top levels becomes an all-zero column
function X = alignCols(T, predNames)
X = zeros(height(T), numel(predNames));
have = ismember(predNames, T.Properties.VariableNames);
X(:, have) = T{:, predNames(have)};
end

%% Helper: fitTuned
%Refits rf or svm with its tuned hyperparameters, same settings as trainModel.m
%Used by the 5x2cv F test and the leak-free CV check
function mdl = fitTuned(name, X, y, p, classNames, predNames)
common = {'ClassNames', classNames, 'PredictorNames', predNames, 'Prior', 'uniform'};
switch name
    case "rf"
        treeArgs = nv(p, ["MinLeafSize", "NumVariablesToSample"]);
        t   = templateTree('Reproducible', true, treeArgs{:});
        ens = nv(p, "NumLearningCycles");
        mdl = fitcensemble(X, y, 'Method', 'Bag', 'Learners', t, ens{:}, common{:});
    case "svm"
        svmArgs = nv(p, ["BoxConstraint", "KernelScale"]);
        t   = templateSVM('KernelFunction', 'gaussian', 'Standardize', false, ...
                          'CacheSize', 'maximal', svmArgs{:});
        mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsone', common{:});
    otherwise
        error('evaluate_models:fitTuned', 'Leak-free refit is only set up for rf and svm.');
end
end

%% Helper: nv
%Builds a name-value list from the fields of p that are listed in names
function args = nv(p, names)
args = {};
for n = names
    if isfield(p, n)
        v = p.(n);
        if iscategorical(v) || isstring(v)
            v = char(v);
        end
        args = [args, {char(n), v}];
    end
end
end