[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 3, 11);
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 3, 11);
normalizedFeatures_train = normalizeFeatures(features_train);
normalizedFeatures_test = normalizeFeatures(features_test);

%scatter3(features(:, 1), features(:, 2), labels)

%one possible attempt of predicting the house condition (this is kind of a
%regression). maybe its better to use categorical labels. or crossentropy +
%softmax.
y = labels_train;
X = [
    normalizedFeatures_train(:, 1:13), ...
    onehotEncode(features_train(:, 14)), ... %zipcode
    normalizedFeatures_train(:, 15:end)
    ];
% setup neural network
inputLayer = Layer(size(X, 2), true, @linear);
outputLayer = Layer(1, false, @linear);
nn = NeuralNetwork(inputLayer, outputLayer);
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.initWeights(0, 5);
nn = nn.train(X, y, .05, 400);
predicted = nn.predict(X)

mean((predicted - y).^2)



% setup neural network
inputLayer = Layer(18, true, @linear);
outputLayer = Layer(5, false, @sigmoid);
nn = NeuralNetwork(inputLayer, outputLayer);
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.addLayer(Layer(20, true, @sigmoid));
nn = nn.initWeights(0, 5);
nn = nn.train(X, y, .05, 111)
predicted = nn.predict(X)

mean((predicted - y).^2)





