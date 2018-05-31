function encoded = onehotEncode(features)
%ONEHOTENCODE Onhot encode the given feature vector

n_categories = length(unique(features));

featureToCategory = containers.Map(unique(features), 1:n_categories); 

encoded = zeros(length(features), n_categories);

for i = 1:length(features)
    encoded(i, featureToCategory(features(i))) = 1;
end
