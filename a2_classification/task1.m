close all; clc;
%% load data
%the first two columns of the csv file (id and date) were removed
%beforehand to make it easier to load/read
[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 7); %col 7 is waterfront (label)
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 7);

%% analyze dataset
n_pos_train = size(find(labels_train == 1), 1)
n_neg_train = size(find(labels_train == 0), 1)
n_pos_test = size(find(labels_test == 1), 1)
n_neg_test = size(find(labels_test == 0), 1)


%% feature selection
% first option: using all features as is (+ bias terms)
X = [
    ones(size(features_train, 1), 1), ...
    normalizeFeatures(features_train)
    ];
Xtest = [
    ones(size(features_test, 1), 1), ...
    normalizeFeatures(features_test)
    ];
% second option: using squared and cubic features as well
X = [
    ones(size(features_train, 1), 1), ...
    normalizeFeatures(features_train), ...
    normalizeFeatures(features_train.^2), ...
    normalizeFeatures(features_train.^3)
    ];
Xtest = [
    ones(size(features_test, 1), 1), ...
    normalizeFeatures(features_test), ...
    normalizeFeatures(features_test.^2), ...
    normalizeFeatures(features_test.^3)
    ];
% third option: zipcode one-hot encoded
X = [
    ones(size(features_train, 1), 1), ...
    normalizeFeatures(features_train(:, 1:13)), ...
    normalizeFeatures(features_train(:, 15:end)), ...
    onehotEncode(features_train(:, 14)) %zipcode
    ];
Xtest = [
    ones(size(features_test, 1), 1), ...
    normalizeFeatures(features_test(:, 1:13)), ...
    normalizeFeatures(features_test(:, 15:end)), ...
    onehotEncode(features_test(:, 14)) %zipcode
    ];
% forth option: only features that are meaningful according to gut feeling
% --> not a good result
% X = [
%     ones(size(features_train, 1), 1), ...
%     normalizeFeatures(features_train(:, 1)), ... %price
%     normalizeFeatures(features_train(:, 9)), ... %grade
%     normalizeFeatures(features_train(:, 15:16)), ... %lat & long
%     onehotEncode(features_train(:, 14)) %zipcode
%     ];
% Xtest = [
%     ones(size(features_test, 1), 1), ...
%     normalizeFeatures(features_test(:, 1)), ... %price
%     normalizeFeatures(features_test(:, 9)), ... %grade
%     normalizeFeatures(features_test(:, 15:16)), ... %lat & long
%     onehotEncode(features_test(:, 14)) %zipcode
%     ];

y = labels_train;
ytest = labels_test;

%% split data
[Xtrain, Xval, ~, ytrain, yval, ~] = splitDataset(X, y, .9, .1);

%% fit model and find good parameters
initialTheta = zeros(1, size(X, 2)); %weights
alpha = 20; %learning rate
n_iters = 10000;
lambdaValues = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
trainLoss = [];
valLoss = [];
testLoss = [];
for lambda = lambdaValues  % visualize results using different lambdas
    [theta, costHistory] = gradientDescent(Xtrain, ytrain, initialTheta, alpha, lambda, n_iters, true);
    trainLoss = [trainLoss, logCost(Xtrain, theta, ytrain, 0)]; % use lambda=0 here because we want the bare cost
    valLoss = [valLoss, logCost(Xval, theta, yval, 0)]; % dito
    testLoss = [testLoss, logCost(Xtest, theta, ytest, 0)]; % dito
end
figure;
plot(lambdaValues, trainLoss, lambdaValues, valLoss, lambdaValues, testLoss);
legend('Training loss', 'Validation loss', 'Test loss');
title('Choosing lambda');
xlabel('lambda');
ylabel('loss');

%% fit model using best parameters and evaluate it
lambda = 0; % regularization parameter, set according to best lambda found in plot before
[theta, costHistory] = gradientDescent(X, y, initialTheta, alpha, lambda, n_iters, true); 

fprintf('Evaluation on training set:\n');
evaluate(predict(Xtrain, theta, true), ytrain, true);
fprintf('Evaluation on validation set:\n');
evaluate(predict(Xval, theta, true), yval, true);
fprintf('Evaluation on test set:\n');
evaluate(predict(Xtest, theta, true), ytest, true);