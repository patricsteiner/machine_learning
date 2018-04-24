close all; clc;
% first row in dataset has been removed (was id --> unnecessary)
[features, labels, featureTitles] = loadCSVData('dev_data.csv', 3); % labels in col 3
X = [
    normalizeFeatures(features)
    ];
y = labels;
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, .8, 0);

% setup neural network
inputLayer = Layer(size(X, 2), true, @linear);
outputLayer = Layer(1, false, @sigmoid);
nn = NeuralNetwork(inputLayer, outputLayer);
nn = nn.addLayer(Layer(size(X, 2), true, @sigmoid));
nn = nn.addLayer(Layer(size(X, 2), true, @sigmoid));
nn = nn.initWeights(-.5, .5);
alpha = .5; %learning rate
n_iters = 50000;
nn = nn.train(Xtrain, ytrain, alpha, n_iters);

predicted = nn.predict(Xtest);
mean((predicted - ytest).^2)






