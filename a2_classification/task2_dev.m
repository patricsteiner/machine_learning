close all; clc;
%% load data
% first row in dataset has been removed (was id --> unnecessary)
[features, labels, featureTitles] = loadCSVData('dev_data.csv', 3); % labels in col 3

%% feature selection
X = [
    normalizeFeatures(features)%, ...
    %normalizeFeatures(features.^2), ...
    %normalizeFeatures(features.^3), ...
    %normalizeFeatures(features.^4)
    ];
y = labels;

%% split data
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .6, .2);

%% setup neural network
inputLayer = Layer(size(X, 2), true, @linear);
outputLayer = Layer(1, false, @sigmoid);
nn = NeuralNetwork(inputLayer, outputLayer);
nn = nn.addLayer(Layer(size(X, 2)*2, true, @sigmoid));
nn = nn.initWeights(-1, 1);

%% set parameters and train network using different lambdas
alpha = 1; %learning rate
n_iters = 1000;

lambdaValues = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
trainLoss = [];
valLoss = [];
testLoss = [];
for lambda = lambdaValues  % visualize results using different lambdas
    nn = nn.initWeights(-1, 1);
    nn = nn.train(Xtrain, ytrain, alpha, lambda, n_iters, true);
    trainLoss = [trainLoss, nn.cost(nn.predict(Xtrain, false), ytrain)];
    valLoss = [valLoss, nn.cost(nn.predict(Xval, false), yval)];
    testLoss = [testLoss, nn.cost(nn.predict(Xtest, false), ytest)];
end
figure;
plot(lambdaValues, trainLoss, lambdaValues, valLoss, lambdaValues, testLoss);
legend('Training loss', 'Validation loss', 'Test loss');
title('Choosing lambda');
xlabel('lambda');
ylabel('loss');

lambda = 0.1; %regularization parameter, choose best according to the visual evaluation above
nn = nn.initWeights(-1,1);
nn = nn.train(Xtrain, ytrain, alpha, lambda, n_iters, true);

%% evaluate
predicted = nn.predict(Xtest, true);
evaluate(predicted, ytest, true);
mean((predicted - ytest).^2);







