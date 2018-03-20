function normalizedFeatures = normalizeFeatures(features)
%NORMALIZEFEATURES Scale and features to avoid overflows and speed up
%gradient descent

normalizedFeatures = zeros(size(features));

for i = 1:size(features, 2)
    normalizedFeatures(:, i) = (features(:, i) - mean(features(:, i))) / std(features(:, i));
end

end

