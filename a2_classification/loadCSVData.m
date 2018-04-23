function [features, labels, featureTitles] = loadCSVData(filename, startcol, labelcol)
%LOADDATA 

features = csvread(filename, 1, startcol-1); % start from 3rd column (price) and 2nd row (skip header)
labels = features(:, labelcol-startcol+1);
featurecols = 1:size(features, 2);
featurecols = featurecols(featurecols ~= labelcol-1);
features = features(:, featurecols);
file = fopen(filename);
featureTitles = split(replace(fgetl(file), '_', ' '), ',');
fclose(file);
featureTitles(labelcol) = [];
if startcol > 1
    featureTitles(1:startcol - 1) = [];
end

end

