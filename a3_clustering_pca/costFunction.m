function cost = costFunction(X, y, centroids)
%COSTFUNCTION sum of squared distances of examples to centroids

m = size(X, 1);

diff = X - centroids(y, :);
cost = 1/m * sum(sum(diff.^2, 2));

end

