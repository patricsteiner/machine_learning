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
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .8, 0);

% setup neural network
inputLayer = Layer(size(X, 2), true, @linear);
outputLayer = Layer(1, false, @sigmoid);
nn = NeuralNetwork(inputLayer, outputLayer);
%nn = nn.addLayer(Layer(size(X, 2)*10, true, @sigmoid));
%nn = nn.addLayer(Layer(size(X, 2)*5, true, @sigmoid));
nn = nn.initWeights(-1, 1);
alpha = .5; %learning rate
lambda = 1; %regularization parameter
n_iters = 10000;
nn = nn.train(Xtrain, ytrain, alpha, lambda, n_iters);

predicted = nn.predict(Xtest);
mean((predicted - ytest).^2)
predicted = nn.predict(Xtrain);
mean((predicted - ytrain).^2)





