function X = applyPrep(T, params, cfg)
%APPLYPREP Transform a table with parameters fitted by FITPREP.
%   X = APPLYPREP(T, PARAMS, CFG) returns a numeric table with named
%   columns: scaled continuous features, 0/1 binary features, and one-hot
%   columns for each categorical feature.

%% Continuous
%Applies the training median, log1p, capping bounds and scaling in the same order as fitPrep
%Values are clamped at 0 before log1p so an unseen negative test value can't break it
Xc = T{:, cellstr(cfg.continuous)};
Xc = fillmissing(Xc, 'constant', params.med);
Xc(:, params.logCols) = log1p(max(Xc(:, params.logCols), 0));
Xc = min(max(Xc, params.lo), params.hi);
Xc = normalize(Xc, 'center', params.C, 'scale', params.S);
X  = array2table(Xc, 'VariableNames', cellstr(cfg.continuous));

%% Binary
%Fills missing values with the training mode, no scaling
if ~isempty(cfg.binary)
    B = fillmissing(T{:, cellstr(cfg.binary)}, 'constant', params.binMode);
    X = [X, array2table(B, 'VariableNames', cellstr(cfg.binary))];
end

%% Categorical
%One-hot encodes over the levels learned on the training set
%Rare or unseen levels go to "Other", missing values go to "Unknown"
for i = 1:numel(cfg.categorical)
    name = cfg.categorical(i);
    lv   = params.levels{i};
    cats = [lv; "Other"; "Unknown"];

    s = toLevelString(T.(name));
    s(ismissing(s) | s == "") = "Unknown";
    s(~ismember(s, cats)) = "Other";

    [~, loc] = ismember(s, cats);
    D = zeros(numel(s), numel(cats));
    D(sub2ind(size(D), (1:numel(s))', loc)) = 1;

    colNames = matlab.lang.makeValidName(name + "_" + cats);
    X = [X, array2table(D, 'VariableNames', cellstr(colNames'))];
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