function [Xtrain, Xtest, ytrain, ytest] = trainTestSplit(X, y)
%TRAINTESTSPLIT Split features and labels into random train and test set (80%/20%)

m = size(X, 1);
shuffledIndexes = randperm(m);
splitAt = floor(m*.8);
Xtrain = X(shuffledIndexes(1:splitAt), :);
ytrain = y(shuffledIndexes(1:splitAt));
Xtest = X(shuffledIndexes(splitAt+1:end), :);
ytest = y(shuffledIndexes(splitAt+1:end));

end

