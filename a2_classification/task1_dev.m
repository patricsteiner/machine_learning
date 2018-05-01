close all; clc;
%% load data
% first row in dataset has been removed (was id --> unnecessary)
[features, labels, featureTitles] = loadCSVData('dev_data.csv', 3); % labels in col 3

%% visualize dataset
pos = find(labels == 1);
neg = find(labels == 0);
plot(features(pos, 1), features(pos, 2), 'bo', 'MarkerSize', 7);
hold on;
plot(features(neg, 1), features(neg, 2), 'rx', 'MarkerSize', 7);
title('dev\_data visualized');
xlabel('x1');
ylabel('x2');
legend('positive', 'negative');

%% feature selection
X = [
    ones(size(features, 1), 1), ...
    features%, ...
    %normalizeFeatures(features.^2)%, ...
    %normalizeFeatures(features.^3), ...
    %normalizeFeatures(features.^4) ,...
    %normalizeFeatures(features(:, 1).*features(:, 2))
    ];
y = labels;

%% split data
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .6, .2);

%% fit model and find good parameters
initialTheta = zeros(1, size(X, 2)); %weights
alpha = .1; %learning rate
n_iters = 1000;
lambdaValues = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
trainLoss = [];
valLoss = [];
testLoss = [];
for lambda = lambdaValues  % visualize results using different lambdas
    [theta, costHistory] = gradientDescent(Xtrain, ytrain, initialTheta, alpha, lambda, n_iters, false);
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
lambda = 0.03; % regularization parameter, set according to best lambda found in plot before
[theta, costHistory] = gradientDescent(Xtrain, ytrain, initialTheta, alpha, lambda, n_iters, true); 

fprintf('Evaluation on training set:\n');
evaluate(predict(Xtrain, theta, true), ytrain, true);
fprintf('Evaluation on validation set:\n');
evaluate(predict(Xval, theta, true), yval, true);
fprintf('Evaluation on test set:\n');
evaluate(predict(Xtest, theta, true), ytest, true);