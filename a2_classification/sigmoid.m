function s = sigmoid(x, derivative)
%SIGMOID Computes the sigmoid function or its derivative if derivative = true

if derivative
    s = sigmoid(x, false) .* (1 - sigmoid(x, false));
else
    s = 1 ./ (1 + exp(-x));
end

end
