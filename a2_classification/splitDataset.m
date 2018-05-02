function [Xtrain, Xval, Xtest, ytrain, yval, ytest] = splitDataset(X, y, trainRatio, validationRatio)
%SPLITDATASET Split features and labels into random train, validation and test set

m = size(X, 1);
shuffledIndexes = randperm(m);
split1 = floor(m * trainRatio);
split2 = floor(m * validationRatio) + split1;
Xtrain = X(shuffledIndexes(1:split1), :);
ytrain = y(shuffledIndexes(1:split1), :);
Xval   = X(shuffledIndexes(split1+1:split2), :);
yval   = y(shuffledIndexes(split1+1:split2), :);
Xtest  = X(shuffledIndexes(split2+1:end), :);
ytest  = y(shuffledIndexes(split2+1:end), :);

end

