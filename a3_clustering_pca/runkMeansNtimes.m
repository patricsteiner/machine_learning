function [centroids, idx, cost] = runkMeansNtimes(X, K, N, max_iters, plot_progress)
%RUNKMEANSNTIMES runs kmeans N times with K clsuters and picks the best result

minCost = inf;
bestCentroids = [];
bestIdx = [];
for i = 1:N
    [centroids, idx, cost] = runkMeans(X, K, max_iters, plot_progress);
    if cost < minCost
        minCost = cost
        bestCentroids = centroids;
        bestIdx = idx;
    end
end

end
