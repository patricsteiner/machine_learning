[features, labels, featureTitles] = loadCSVData('dev_data.csv', 2, 4);

scatter3(features(:, 1), features(:, 2), labels)

m = size(features, 1);
X = [ones(m, 1), features];
y = labels;
theta = zeros(1, 3);

[theta, J_history] = gradientDescent(X, y, theta, .1, .5, 1000)

evaluate(predict(X, theta, true), y)