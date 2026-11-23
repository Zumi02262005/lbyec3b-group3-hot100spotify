%% PLOT CLASS DISTRIBUTION - Billboard Hot 100 & Spotify Audio Features (LBYEC3B Team 3)
%Bar chart of chart_success_tier counts in the training and test sets
%Shows the class imbalance that motivates macro F1 and Prior = uniform
%Needs data/processed/preprocessed.mat from main.m
clear; clc; close all;

%% Load Data
%Only the targets are needed
S = load(fullfile('data', 'processed', 'preprocessed.mat'), 'y_train', 'y_test');
y_train = S.y_train;
y_test  = S.y_test;

%% Class Order and Counts
%Tiers ordered low to high, the same order used in every confusion matrix
tiers     = ["hot_100_entry", "top_forty_hit", "top_ten_smash", "historic_number_one"];
dispNames = strrep(tiers, "_", " ");
nTrain = arrayfun(@(t) nnz(string(y_train) == t), tiers);
nTest  = arrayfun(@(t) nnz(string(y_test) == t), tiers);
pTrain = 100 * nTrain / sum(nTrain);
pTest  = 100 * nTest / sum(nTest);

counts = table(dispNames', nTrain', pTrain', nTest', pTest', ...
    'VariableNames', {'Tier', 'TrainCount', 'TrainPct', 'TestCount', 'TestPct'});
disp(counts)

%% Figure: Class Distribution
%Grouped bars for train and test, each bar labeled with its count and percentage
fig = figure('Name', 'Class distribution', 'Position', [100 100 800 480]);
b = bar(1:numel(tiers), [nTrain; nTest]', 'grouped');
xticks(1:numel(tiers));
xticklabels(dispNames);
xlabel('Chart success tier');
ylabel('Number of songs');
legend(sprintf('Training set (n = %d)', sum(nTrain)), sprintf('Test set (n = %d)', sum(nTest)), ...
    'Location', 'northeast');
title('Class distribution of chart\_success\_tier');

labelsTrain = compose("%d\n(%.1f%%)", nTrain', pTrain');
labelsTest  = compose("%d\n(%.1f%%)", nTest', pTest');
text(b(1).XEndPoints, b(1).YEndPoints, labelsTrain, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
text(b(2).XEndPoints, b(2).YEndPoints, labelsTest, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8);
ylim([0, 1.15 * max(nTrain)]);

%% Save
%Same folder as the other result figures
figDir = fullfile('results', 'figures');
if ~isfolder(figDir)
    mkdir(figDir);
end
exportgraphics(fig, fullfile(figDir, 'class_distribution.png'), 'Resolution', 200);