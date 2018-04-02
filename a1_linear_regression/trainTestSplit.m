function [Xtrain, Xtest, ytrain, ytest] = trainTestSplit(X, y, trainRatio)
%TRAINTESTSPLIT Split features and labels into random train and test set
%(trainRatio for train part, 1-trainRatio for test part)

m = size(X, 1);
shuffledIndexes = randperm(m);
splitAt = floor(m * trainRatio);
Xtrain = X(shuffledIndexes(1:splitAt), :);
ytrain = y(shuffledIndexes(1:splitAt));
Xtest = X(shuffledIndexes(splitAt+1:end), :);
ytest = y(shuffledIndexes(splitAt+1:end));

end

