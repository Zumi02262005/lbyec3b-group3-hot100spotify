%% MAIN - Billboard Hot 100 & Spotify Audio Features (LBYEC3B Team 3)
%Clears the workspace and sets the seed that every model must share (rng(42))
clear; clc; close all;
rng(42)

%% Data File
%Path to the raw Kaggle CSV, relative to the project root
dataFile = fullfile('data', 'raw', 'billboard_hot_100_spotify_audio_features_enriched.csv');

%% Task and Target
%Multiclass classification on chart_success_tier
%For regression instead, set task = "regression" and target = "spotify_track_popularity"
%makeTarget can hold a function that derives the target, e.g. @(T) T.peak_position <= 10
cfg.task       = "classification";
cfg.target     = "chart_success_tier";
cfg.makeTarget = [];

%% Feature Columns
%idVars identifies duplicate songs
%continuous features get imputation, log transform, capping and scaling
%binary features only get missing values filled
%categorical features get one-hot encoded
cfg.idVars      = "song_id";
cfg.continuous  = ["spotify_track_duration_ms", "danceability", "energy", "loudness", ...
                   "speechiness", "acousticness", "instrumentalness", "liveness", ...
                   "valence", "tempo"];
cfg.binary      = ["mode", "spotify_track_explicit"];
cfg.categorical = ["key", "time_signature", "musical_era"];

%% Excluded Columns
%IDs/text:  song_id, performer, song, spotify_track_id, spotify_track_preview_url, spotify_track_album
%Leakage:   peak_position, weeks_on_chart, first_week, last_week, hit_longevity_profile
%Generated: evidence_span, cot_rationale, confidence_score, acoustic_mood_cluster

%% Genre Families
%spotify_genre holds lists like "['dance pop', 'pop']", so each list is mapped to
%broad family flags (genre_pop, genre_rap, ...). A family matches if ANY genre in
%the list contains one of its keywords, so a song can belong to several families
cfg.genreVar = "spotify_genre";
cfg.genreFamilies = struct( ...
    'pop',        "pop", ...
    'rap',        ["rap", "hip hop", "trap", "drill"], ...
    'rock',       ["rock", "metal", "punk", "grunge", "indie"], ...
    'rnb_soul',   ["r&b", "soul", "funk", "motown", "urban contemporary", ...
                   "quiet storm", "new jack swing", "rare groove", "doo-wop"], ...
    'country',    ["country", "bluegrass", "americana", "folk", "western", "nashville"], ...
    'dance_elec', ["dance", "edm", "house", "disco", "electro", "techno", "freestyle", "hi-nrg"], ...
    'latin',      ["latin", "reggaeton", "norteno", "duranguense", "banda", "mexican", ...
                   "salsa", "bachata"], ...
    'reggae',     ["reggae", "dancehall", "ska"], ...
    'easy_listening', ["adult standards", "easy listening", "lounge", "new age", ...
                       "instrumental", "soundtrack", "show tunes", "broadway", "hollywood"], ...
    'jazz_blues', ["jazz", "blues", "swing"]);

%% Cleaning and Transform Settings
%positiveOnly: values <= 0 in these columns are data-entry errors and become missing
%skewThresh: features with skewness above this get a log1p transform
%maxLevels: number of most frequent levels kept per categorical feature
%Each value must be justified from the EDA
cfg.positiveOnly = ["tempo", "spotify_track_duration_ms"];
cfg.skewThresh   = 1;
cfg.maxLevels    = 15;

%% Split Settings
%Stratified 80/20 hold-out test set, then stratified 5-fold CV on the training set
cfg.testFrac = 0.2;
cfg.kFolds   = 5;

%% Data Preprocessing
%Runs the full pipeline and unpacks the results
%cvKfold must be passed to every model through 'CVPartition'
%cfg is replaced so it also lists the generated genre_* columns
D = preprocessData(dataFile, cfg);

X_train = D.Xtrain;
y_train = D.ytrain;
X_test  = D.Xtest;
y_test  = D.ytest;
cvKfold = D.cvKfold;
params  = D.params;
cfg     = D.cfg;

%% Sanity Checks
%Stops if any missing values remain and prints the training class balance
assert(nnz(ismissing(X_train)) == 0, 'Missing values remain in X_train');
assert(nnz(ismissing(X_test))  == 0, 'Missing values remain in X_test');

if cfg.task == "classification"
    disp('Class balance (training set):');
    tabulate(y_train)
end

%% Save Preprocessed Data
%Saves everything the modeling scripts need so preprocessing doesn't have to be rerun
save(fullfile('data', 'processed', 'preprocessed.mat'), ...
     'X_train', 'y_train', 'X_test', 'y_test', 'cvKfold', 'params', 'cfg');