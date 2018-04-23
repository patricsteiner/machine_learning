[features_train, labels_train, featureTitles] = loadCSVData('house_train_data.csv', 3, 9);
[features_test, labels_test, featureTitles] = loadCSVData('house_test_data.csv', 3, 9);
normalizedFeatures_train = normalizeFeatures(features_train);
normalizedFeatures_test = normalizeFeatures(features_test);

%scatter3(features(:, 1), features(:, 2), labels)

[m, n] = size(features_train);
X = [ones(m, 1), normalizedFeatures_train(:, 1:4)];
y = labels_train;
theta = ones(1, size(X, 2));

[theta, J_history] = gradientDescent(X, y, theta, .1, .5, 1000)

[m, n] = size(features_test);
X = [ones(m, 1), normalizedFeatures_test(:, 1:4)];
y = labels_test;
evaluate(predict(X, theta, true), y)