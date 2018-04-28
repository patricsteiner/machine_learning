close all; clc;
%the first two columns of the csv file (id and date) were removed
%beforehand to make it easier to load/read
[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 7); %col 7 is waterfront (label)
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 7);

X = [
    ones(size(features_train, 1), 1), ...
    onehotEncode(features_train(:, 6)), ... %floors
    onehotEncode(features_train(:, 8)), ... %condition
    onehotEncode(features_train(:, 14)), ... %zipcode
    normalizeFeatures(features_train)%(:, 15:16) %long & lat
    ];
y = labels_train;
Xtest = [
    ones(size(features_test, 1), 1), ...
    onehotEncode(features_test(:, 6)), ... %floors
    onehotEncode(features_test(:, 8)), ... %condition
    onehotEncode(features_test(:, 14)), ... %zipcode
    normalizeFeatures(features_test)%(:, 15:16) %long & lat
    ];
ytest = labels_test;
[Xtrain, Xval, ~, ytrain, yval, ~] = splitDataset(X, y, .9, .1);

initialTheta = zeros(1, size(X, 2)); %weights

alpha = 1; %learning rate
n_iters = 111;
bestFscore = 0;
bestLambda = -111110;
bestTheta = initialTheta;
lambdaValues = [-100, -10, -1, -0.1, 0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
trainLoss = [];
valLoss = [];
for lambda = lambdaValues  % find best regularization param
    [theta, J_history] = gradientDescent(Xtrain, ytrain, initialTheta, alpha, lambda, n_iters);
    trainLoss = [trainLoss, logCost(Xtrain, theta, ytrain, 0)]; % use lambda=0 here because we want the bare cost
    valLoss = [valLoss, logCost(Xval, theta, yval, 0)]; % dito
    fscore = evaluate(predict(Xval, theta, true), yval, false)
    if fscore > bestFscore
        bestFscore = fscore;
        bestLambda = lambda;
        bestTheta = theta;
    end
end
figure;
plot(lambdaValues, trainLoss, lambdaValues, valLoss);
legend('Training loss', 'Validation loss');
title('Choosing lambda');
xlabel('lambda');
ylabel('loss');
fprintf('Best lambda  = %d\n', bestLambda);
fprintf('Best F-score = %.3f\n', bestFscore);

fprintf('Evaluation on training(+validation) set:\n');
evaluate(predict(X, bestTheta, true), y, true)
fprintf('Evaluation on test set:\n');
evaluate(predict(Xtest, bestTheta, true), ytest, true)