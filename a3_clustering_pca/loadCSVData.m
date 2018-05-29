function [features, featureTitles] = loadCSVData(filename)
%LOADDATA load values from csv files. Assumes file to only contain
%numerical values.

features = csvread(filename, 1, 0); %skip first row (headers)
file = fopen(filename);
featureTitles = split(replace(fgetl(file), '_', ' '), ',');
fclose(file);

end

