function J = cost(X, y, theta)
%COST Compute cost for linear regression
%   J = COST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);
h = X * theta;
J = 1/(2*m) * sum((h - y).^2);

end