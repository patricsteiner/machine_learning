function fscore = evaluateMultiClass(predicted, actual, print)
%EVALUATE prints confusion matrix, accuracy, precision, recall and f-score
% of a multiclass prediction ("one-hot encoded")

confusionMatrix = zeros(size(predicted, 2));
for n = 1:size(predicted, 1)
    i = find(predicted == 1);
    j = find(actual == 1);
	confusionMatrix(i, j) = confusionMatrix(i, j) + 1;
end
    
tp = sum(predicted & actual);
fp = sum(predicted & ~actual);
tn = sum(~predicted & ~actual);
fn = sum(~predicted & actual);
precision = tp / (tp + fp);
recall = tp / (tp + fn);
accuracy = (tp + tn) / (tp + fp + tn + fn);
fscore = 2 * (precision * recall) / (precision + recall);

if print
    fprintf('Confusion Matrix:\n');
    disp(confusionMatrix);
    fprintf('Accuracy:\t%.3f\n', accuracy);
    fprintf('Precision:\t%.3f\n', precision);
    fprintf('Recall: \t%.3f\n', recall);
    fprintf('F-Score:\t%.3f\n', fscore);
end
    
end

