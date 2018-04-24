function [features, labels, featureTitles] = loadCSVData(filename, labelcol)
%LOADDATA load values from csv files. Assumes file to only contain
%numerical values.

content = csvread(filename, 1, 0); %skip first row (headers)
labels = content(:, labelcol);
features = [content(:, 1:labelcol-1), content(:, labelcol+1:end)];
file = fopen(filename);
featureTitles = split(replace(fgetl(file), '_', ' '), ',');
fclose(file);
featureTitles(labelcol) = [];

end

