clear; clc; close all;
rng(100)

%% Data Preprocessing
[X_train, X_test, cv, params] = prepData('billboard100.csv');