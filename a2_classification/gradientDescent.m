function [theta, costHistory] = gradientDescent(X, y, theta, alpha, lambda, n_iters, visualize)
%GRADIENTDESCENT Performs gradient descent to learn optimal theta values

costHistory = zeros(n_iters, 1);

for i = 1:n_iters
    [cost, grad] = logCost(X, theta, y, lambda);
    costHistory(i) = cost;
    theta = theta - alpha * grad;
end

if visualize
    figure;
    plot(1:n_iters, costHistory);
    title('Gradient descent');
    xlabel('iteration');
    ylabel('logCost');
end

end
