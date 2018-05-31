function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
idx = zeros(size(X,1), 1);
m = size(X, 1); % number of training examples

for i = 1:m
    minDistance = inf;
    for k = 1:K
        distance = norm(X(i, :) - centroids(k, :));
        if (distance < minDistance)
            minDistance = distance;
            closestCentroid = k;
        end
    end
    idx(i) = closestCentroid;
end

end

