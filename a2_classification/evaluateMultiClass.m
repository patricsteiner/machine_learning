function fscore = evaluateMultiClass(predicted, actual, print)
%EVALUATE prints confusion matrix, accuracy, precision, recall and f-score
% of a multiclass prediction ("one-hot encoded")

confusionMatrix = zeros(size(predicted, 2));
for n = 1:size(predicted, 1)
    i = find(actual(n, :) == 1);
    j = find(predicted(n, :) == 1);
	confusionMatrix(i, j) = confusionMatrix(i, j) + 1;
end

% calculate values per class (result is row vector)
tp = sum(predicted & actual);
fp = sum(predicted & ~actual);
tn = sum(~predicted & ~actual);
fn = sum(~predicted & actual);

precision = tp ./ (tp + fp);
recall = tp ./ (tp + fn);
accuracy = (tp + tn) ./ (tp + fp + tn + fn);
fscore = 2 * (precision .* recall) ./ (precision + recall);
% make sure the averages are not messed up by potential NaNs.
precision(isnan(fscore)) = 0; 
recall(isnan(fscore)) = 0;
fscore(isnan(fscore)) = 0; 

% now instead of taking the mean, calculate a weighted average for all the
% metrics, this will give more meaningful results.
total_per_class = sum(actual);
total = sum(total_per_class);
precision = sum(precision .* total_per_class) / total;
recall = sum(recall .* total_per_class) / total;
accuracy = sum(accuracy .* total_per_class) / total;
fscore = sum(fscore .* total_per_class) / total;

if print
    fprintf('Confusion Matrix:\n');
    disp(confusionMatrix);
    fprintf('Accuracy:\t%.3f\n', accuracy);
    fprintf('Precision:\t%.3f\n', precision);
    fprintf('Recall: \t%.3f\n', recall);
    fprintf('F-score:\t%.3f\n', fscore);
end
    
end

