function [J, grad] = logCost(X, theta, y, lambda)
%LOGCOST Compute cost and gradient for logistic regression with regularization

m = length(y);
h = predict(X, theta, false);
squaredThetas = theta.^2;
%not regularizing the bias term
J = 1/m * (-y' * log(h) - (1-y)' * log(1-h)) + lambda/(2*m) * sum(squaredThetas(2:end));
grad = 1/m * (h-y)' * X  + lambda/m * [0, theta(2:end)];

end
