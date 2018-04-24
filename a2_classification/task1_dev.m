close all; clc;
[features, labels, featureTitles] = loadCSVData('dev_data.csv', 2, 4); % start from col 2, labels in col 4
normalizedFeatures = normalizeFeatures(features);
X = [
    ones(size(features, 1), 1), ...
    normalizedFeatures(:, 1:2), ...
    %normalizedFeatures(:, 1).^2, ...
    %normalizedFeatures(:, 2).^2, ...
    ];
y = labels;
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .6, .2);

theta = zeros(1, size(X, 2)); %weights

alpha = .1; %learning rate
n_iters = 1000;
lambda = 11;
bestFscore = 0;
bestLambda = 0;
lambdaValues = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];
trainLoss = [];
valLoss = [];
for lambda = lambdaValues  % find best regularization param
    [theta, J_history] = gradientDescent(Xtrain, ytrain, theta, alpha, lambda, n_iters);
    trainLoss = [trainLoss, logCost(Xtrain, theta, ytrain, 0)]; % use lambda=0 here because we want the bare cost
    valLoss = [valLoss, logCost(Xval, theta, yval, 0)]; % dito
    fscore = evaluate(predict(Xval, theta, true), yval, false);
    if fscore > bestFscore
        bestFscore = fscore;
        bestLambda = lambda;
    end
end
plot(lambdaValues, trainLoss, lambdaValues, valLoss);
legend('Training loss', 'Validation loss');
title('Choosing lambda');
xlabel('lambda');
ylabel('loss');
fprintf('Best lambda  = %d\n', bestLambda);
fprintf('Best F-score = %.3f\n', bestFscore);

fprintf('Evaluation on test set:\n');
evaluate(predict(Xtest, theta, true), ytest, true)