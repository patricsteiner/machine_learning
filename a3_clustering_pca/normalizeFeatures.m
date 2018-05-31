function normalizedFeatures = normalizeFeatures(features)
%NORMALIZEFEATURES Scale the features to be in a reasonable range and make
%computations easier

normalizedFeatures = zeros(size(features));

for i = 1:size(features, 2)
    normalizedFeatures(:, i) = (features(:, i) - mean(features(:, i))) / std(features(:, i));
end

end

