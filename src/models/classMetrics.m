function M = classMetrics(yTrue, yPred, classNames, scores)
%CLASSMETRICS Classification metrics for one set of predictions.
%   M = CLASSMETRICS(YTRUE, YPRED, CLASSNAMES) returns accuracy, macro F1,
%   weighted F1, balanced accuracy, per-class precision/recall/F1 and the
%   confusion matrix, with rows and columns in the order of CLASSNAMES.
%
%   M = CLASSMETRICS(..., SCORES) also returns the macro one-vs-rest
%   ROC-AUC. SCORES must have one column per class in CLASSNAMES order.
%
%   A class that is never predicted gets precision 0 (and F1 0), so a
%   model that ignores a minority class is penalized in macro F1.

%% Confusion Matrix
%Rows are true classes, columns are predicted classes
C  = confusionmat(yTrue, yPred, 'Order', classNames);
tp = diag(C);

%% Per-Class Precision, Recall and F1
%Zero denominators become 0 instead of NaN
predCount = sum(C, 1)';
trueCount = sum(C, 2);

precision = zeros(size(tp));
recall    = zeros(size(tp));
f1        = zeros(size(tp));

hasPred = predCount > 0;
hasTrue = trueCount > 0;
precision(hasPred) = tp(hasPred) ./ predCount(hasPred);
recall(hasTrue)    = tp(hasTrue) ./ trueCount(hasTrue);

hasBoth = (precision + recall) > 0;
f1(hasBoth) = 2 * precision(hasBoth) .* recall(hasBoth) ./ (precision(hasBoth) + recall(hasBoth));

%% Summary Metrics
%Macro F1 is the primary metric, the others are supporting metrics
M.accuracy    = sum(tp) / sum(C(:));
M.macroF1     = mean(f1);
M.weightedF1  = sum(f1 .* trueCount) / sum(trueCount);
M.balancedAcc = mean(recall(hasTrue));
M.precision   = precision';
M.recall      = recall';
M.f1          = f1';
M.confusion   = C;

%% ROC-AUC
%Macro average of the one-vs-rest AUC of each class, only if scores are given
M.macroAUC = NaN;
if nargin >= 4 && ~isempty(scores)
    rc = rocmetrics(yTrue, scores, classNames);
    M.macroAUC = mean(rc.AUC);
end
end