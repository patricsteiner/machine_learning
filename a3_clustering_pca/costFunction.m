function cost = costFunction(X, y, centroids)
%COSTFUNCTION sum of squared distances of examples to centroids

m = size(X, 1);

cost = 0;
for i = 1:m
    distance = sum((X(i, :) - centroids(y(i))).^2);
    cost = cost + 1/m * distance;
end

end

