close all; clc;
%% load data
%the first two columns of the csv file (id and date --> non-features) were removed
%beforehand to make it easier to load/read
[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 9); %col 9 is condition (label)
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 9);

%% analyze/visualize data
figure;
hist(labels_train);
title('histogram of train labels');
xlabel('condition');
ylabel('frequency');
figure;
hist(labels_test);
title('histogram of test labels');
xlabel('condition');
ylabel('frequency');

%% feature selection
X = [
    normalizeFeatures(features_train(:, 1:13)), ...
    normalizeFeatures(features_train(:, 15:end)), ...
    onehotEncode(features_train(:, 14)) %zipcode
    ];
Y = zeros(size(X, 1), 5);
for i = 1:5
    Y(:, i) = labels_train == i; % make categorical labels
end
Xtest = [
    normalizeFeatures(features_test(:, 1:13)), ...
    normalizeFeatures(features_test(:, 15:end)), ...
    onehotEncode(features_test(:, 14)) %zipcode
    ];
Ytest = zeros(size(Xtest, 1), 5);
for i = 1:5
    Ytest(:, i) = labels_test == i; % make categorical labels
end


%% split data
[Xtrain, Xval, ~, Ytrain, Yval, ~] = splitDataset(X, Y, .9, .1);

%% setup neural network
inputLayer = Layer(size(X, 2), true, @linear);
outputLayer = Layer(size(Y, 2), false, @sigmoid);
nn = NeuralNetwork(inputLayer, outputLayer);
nn = nn.addLayer(Layer(size(X, 2), true, @relu));

%% set parameters and train network using different lambdas
alpha = .1; %learning rate
n_iters = 1000;
init_weight_min = -1; %lower bound for random weight initialization
init_weight_max = 1; % upper bound

% lambdaValues = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100];
% trainLoss = [];
% valLoss = [];
% testLoss = [];
% for lambda = lambdaValues  % visualize results using different lambdas
%     nn = nn.initWeights(init_weight_min, init_weight_max);
%     nn = nn.train(Xtrain, Ytrain, alpha, lambda, n_iters, true);
%     trainLoss = [trainLoss, nn.logCost(nn.predict(Xtrain, false), Ytrain)];
%     valLoss = [valLoss, nn.logCost(nn.predict(Xval, false), Yval)];
%     testLoss = [testLoss, nn.logCost(nn.predict(Xtest, false), Ytest)];
% end
% figure;
% plot(lambdaValues, trainLoss, lambdaValues, valLoss, lambdaValues, testLoss);
% legend('Training loss', 'Validation loss', 'Test loss');
% title('Choosing lambda');
% xlabel('lambda');
% ylabel('loss');

lambda = 1; %regularization parameter, choose best according to the visual evaluation above
n_iters = 100; % potentially increase n_iters here after finding a good lambda
nn = nn.initWeights(init_weight_min, init_weight_max);
nn = nn.train(Xtrain, Ytrain, alpha, lambda, n_iters, true);



%% evaluate
fprintf('Evaluation on training set:\n');
evaluateMultiClass(nn.predict(Xtrain, true), Ytrain, true);
fprintf('\nEvaluation on validation set:\n');
evaluateMultiClass(nn.predict(Xval, true), Yval, true);
fprintf('\nEvaluation on test set:\n');
evaluateMultiClass(nn.predict(Xtest, true), Ytest, true);

