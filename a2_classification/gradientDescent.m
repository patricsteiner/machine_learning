function [theta, J_history] = gradientDescent(X, y, theta, alpha, lambda, n_iters)
%GRADIENTDESCENT Performs gradient descent to learn optimal theta values

J_history = zeros(n_iters, 1);

for i = 1:n_iters
    [J, grad] = logCost(X, theta, y, lambda);
    J_history(i) = J;
    theta = theta - alpha * grad;
end

figure;
plot(1:n_iters, J_history);
title('Gradient descent');
xlabel('iteration');
ylabel('logCost');

end
