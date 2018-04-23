function fscore = evaluate(predicted, actual)
%EVALUATE prints confusion matrix, accuracy, precision, recall and f-score

tp = sum(predicted & actual);
fp = sum(predicted & ~actual);
tn = sum(~predicted & ~actual);
fn = sum(~predicted & actual);
precision = tp / (tp + fp);
recall = tp / (tp + fn);
accuracy = (tp + tn) / (tp + fp + tn + fn);
fscore = 2 * (precision * recall) / (precision + recall);

fprintf('Confusion Matrix:\n');
fprintf('-------------\n');
fprintf('|%5d|%5d|\n', tp, fp);
fprintf('|-----------|\n');
fprintf('|%5d|%5d|\n', fn, tn);
fprintf('-------------\n');
fprintf('Accuracy:\t%.3f\n', accuracy);
fprintf('Precision:\t%.3f\n', precision);
fprintf('Recall: \t%.3f\n', recall);
fprintf('F-Score:\t%.3f\n', fscore);

end

