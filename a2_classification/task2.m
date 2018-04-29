close all; clc;
%the first two columns of the csv file (id and date --> non-features) were removed
%beforehand to make it easier to load/read
[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 9); %col 9 is condition (label)
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 9);

Xtrain = [
   % onehotEncode(features_train(:, 6)), ... %floors
    %onehotEncode(features_train(:, 8)), ... %condition
    %onehotEncode(features_train(:, 14)), ... %zipcode
    normalizeFeatures(features_train)%(:, 15:16) %long & lat
    ];
ytrain = labels_train;
Xtest = [
    %ones(size(features_test, 1), 1), ...
    %onehotEncode(features_test(:, 6)), ... %floors
    %onehotEncode(features_test(:, 8)), ... %condition
    %onehotEncode(features_test(:, 14)), ... %zipcode
    normalizeFeatures(features_test)%(:, 15:16) %long & lat
    ];
ytest = labels_test;

alpha = .005; %learning rate
lambda = 0; %regularization parameter
n_iters = 111;
inputLayer = Layer(size(Xtrain, 2), true, @linear);
outputLayer = Layer(1, false, @linear);
nn = NeuralNetwork(inputLayer, outputLayer);
%nn = nn.addLayer(Layer(20, true, @sigmoid));
%nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.initWeights(-.5, .5);
nn = nn.train(Xtrain, ytrain, alpha, lambda, n_iters);
predictedTrain = nn.predict(Xtrain);
predictedTest = nn.predict(Xtest);
mean((predictedTrain - ytrain).^2);
mean((predictedTest - ytest).^2);

