clear ; close all; clc
%% load & prepare data
[features, featureTitles] = loadCSVData('house_data.csv');

indexes = [1:6. 9:13, 16:19]; % selection of features, avoiding categorical
X = normalizeFeatures(features(:, indexes));
labels = featureTitles(indexes);

%% reduce dimensionality using pca
[U, S] = pca(X);

n_components = 3;
Z = projectData(X, U, n_components);

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 'b.');
title('X projected on first 3 PCs');
xlabel('component 1');
ylabel('component 2');
zlabel('component 3');

%% find good number of clusters using elbow plot
max_iters = 20;
cost_history = [];
N = 20;
for K = 2:20
    [centroids, y, cost] = runkMeansNtimes(Z, K, N, max_iters, false);
    cost_history = [cost_history, cost];
end

figure;
plot(2:20, cost_history)
title('elbow plot of kmeans algorithm')
xlabel('number of clusters')
ylabel('cost function (distortion)')

%% use silhouette plot and values to find best number of clsuters
% according to the elbow plot, values around 6 seem to be a good choice.
% Therefore, further investigate values between 5 and 9 further.
mean_silhouette_history = [];
for K = 5:9
    [centroids, y, cost] = runkMeansNtimes(Z, K, N, max_iters, false);
    figure;
    [sil, ~] = silhouette(Z, y);
    title(sprintf('silhouette plot for K=%d', K));
    mean_silhouette_history = [mean_silhouette_history, mean(sil)];
end

figure;
plot(5:9, mean_silhouette_history)
title('mean silhouette values vs. K')
xlabel('number of clusters')
ylabel('mean silhouette value')

%% run clsutering with K=7 and plot the cluster results
K = 7;
[centroids, y, cost] = runkMeansNtimes(Z, K, N, max_iters, false);

palette = hsv(K + 1);
colors = palette(y, :);

scatter3(Z(:, 1), Z(:, 2), Z(:, 3), 10, colors, 'filled');
title('Clsutering result on first 3 PCs');
xlabel('component 1');
ylabel('component 2');
zlabel('component 3');

% Plot some example data (just for completeness' sake and comparison to
% previous task)
scatter(X(:, 1), X(:, 2), 10, colors, 'filled');
title('clustering result');
xlabel(labels(1));
ylabel(labels(2));
figure;
scatter(X(:, 2), X(:, 3), 10, colors, 'filled');
title('clustering result');
xlabel(labels(2));
ylabel(labels(3));
figure;
scatter(X(:, 2), X(:, 4), 10, colors, 'filled');
title('clustering result');
xlabel(labels(2));
ylabel(labels(4));
figure;
scatter3(X(:, 12), X(:, 13), X(:, 1), 10, colors, 'filled');
title('clustering result');
xlabel(labels(12));
ylabel(labels(13));
zlabel(labels(1));

%% characterize clusters by analysing the variance per feature
for k = 1:K
    cluster_size = sum(y==k) / size(X, 1);
    vars = var(X(y==k, :));
    means = mean(X(y==k, :));
    [~, indices] = sort(vars);
    fprintf('cluster %d contains %.1f%% of the datapoints:\n', k, cluster_size*100);
    for i = indices
        fprintf('\t%-15s: var = %.2f,\tmean = %.2f\n', labels{i}, vars(i), means(i));
    end
    fprintf('\n');
end
