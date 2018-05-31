function centroids = randomCentroids(X, K)
%KMEANSINITCENTROIDS initializes K random cenroids

randidx = randperm(size(X, 1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);

end

