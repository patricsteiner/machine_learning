function [features, labels, featureTitles] = loadData(filename)
%LOADDATA Summary of this function goes here

features = csvread(filename, 3, 2); % start from 3rd column (price) and 2nd row (skip header)
labels = features(:, 1);
features = features(:, 2:end);
file = fopen(filename);
featureTitles = split(replace(fgetl(file), '_', ' '), ',');
featureTitles(1:3) = []; % remove id, date, price labels
fclose(file);

end

