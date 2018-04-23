function h = predict(X, theta, binary)
%PREDICT predict output values using features X and weights theta

h = sigmoid(X * theta', false);
if binary
    h = h >= .5;
end
    
end

