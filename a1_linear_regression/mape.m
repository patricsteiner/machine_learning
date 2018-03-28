function [m, dist] = mape(X, y, theta)
%MAPE mean absolute percentage error

actual = y;
predicted = X * theta;

dist = 100 / length(actual) * abs((actual - predicted) ./ actual);
m = sum(dist);

end

