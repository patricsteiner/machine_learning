function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    h = theta' * X';
    theta = theta - (alpha * 1/m * ((h' - y)' * X))';
    J_history(iter) = cost(X, y, theta);
end

end