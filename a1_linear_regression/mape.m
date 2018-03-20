function [m, dist] = mape(actual, predicted)
%MAPE mean absolute percentage error

dist = 100 / length(actual) * abs((actual - predicted) ./ actual);
m = sum(dist);

end

