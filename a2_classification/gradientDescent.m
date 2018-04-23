function [theta, J_history] = gradientDescent(X, y, theta, alpha, lambda, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn optimal theta values

J_history = zeros(num_iters, 1);

for i = 1:num_iters
    [J, grad] = logCost(X, theta, y, lambda);
    J_history(i) = J;
    theta = theta - alpha * grad;
end

end
