close all; clc;
% first row in dataset has been removed (was id --> unnecessary)
[features, labels, featureTitles] = loadCSVData('dev_data.csv', 3); % labels in col 3
X = [
    ones(size(features, 1), 1), ...
    normalizeFeatures(features), ...
    normalizeFeatures(features.^2), ...
    normalizeFeatures(features.^3), ...
    normalizeFeatures(features.^4)
    ];
y = labels;
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .6, .2);
initialTheta = zeros(1, size(X, 2)); %weights

alpha = 1; %learning rate
n_iters = 100;
bestFscore = 0;
bestLambda = 0;
bestTheta = initialTheta;
lambdaValues = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
trainLoss = [];
valLoss = [];
testLoss = [];
for lambda = lambdaValues  % find best regularization param
    [theta, J_history] = gradientDescent(Xtrain, ytrain, initialTheta, alpha, lambda, n_iters);
    trainLoss = [trainLoss, logCost(Xtrain, theta, ytrain, 0)]; % use lambda=0 here because we want the bare cost
    valLoss = [valLoss, logCost(Xval, theta, yval, 0)]; 
    testLoss = [testLoss, logCost(Xtest, theta, ytest, 0)]; 
    fscore = evaluate(predict(Xval, theta, true), yval, false) %eval on validation set!
    if fscore > bestFscore % TODO or use logcost here?
        bestFscore = fscore;
        bestLambda = lambda;
        bestTheta = theta;
    end
end
figure;
plot(lambdaValues, trainLoss, lambdaValues, valLoss, lambdaValues, testLoss);
legend('Training loss', 'Validation loss', 'Test loss');
title('Choosing lambda');
xlabel('lambda');
ylabel('loss');
fprintf('Best lambda  = %d\n', bestLambda);
fprintf('Best F-score = %.3f\n', bestFscore);

fprintf('Evaluation on training(+validation) set:\n');
evaluate(predict(X, bestTheta, true), y, true)
fprintf('Evaluation on test set:\n');
evaluate(predict(Xtest, bestTheta, true), ytest, true)