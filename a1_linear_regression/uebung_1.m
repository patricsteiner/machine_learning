%% 1) Load and visualize data
housedata = importfile('house_data.csv'); % takes a couple seconds to load all the data in a table
figure('Name', 'Comparison of house features vs. price'); % display scatterplots for all features vs. the price
for i = 4:21 % first 3 columns are id, date and price, features start from column 4
    subplot(4, 5, i-3);
    scatter(housedata{:, i}, housedata.price, 0.5, 'r')
    ylabel('price');
    xlabel(housedata.Properties.VariableNames{i});
    set(gca,'FontSize', 8);
end
set(gcf, 'Position', [200, 200, 1000, 800]);

disp('press any key to continue...');
pause;

%% 1.5) Feature scaling (standardization) to avoid overflows and speed up gradient descent
for i = 4:21 % first 3 columns are id, date and price, features start from column 4
    housedata{:, i} = (housedata{:, i} - mean(housedata{:, i})) / std(housedata{:, i});
end

%% 2) Linear regression for price using sqft_living
m = size(housedata, 1); % number of training samples
X = [ones(m, 1), housedata.sqft_living]; % feature matrix with added column of ones
y = housedata.price; % labels
theta = zeros(2, 1); % initial parameters (weights)
alpha = 0.1; % learning rate
num_iters = 100;
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% visualize gradient descent
figure;
plot(1:num_iters, J_history);
title('Gradient descent visualization');
xlabel('number of iterations');
ylabel('cost');

% visualize regression line and features
figure;
scatter(X(:, 2), y, 1, '.');
hold on;
plot(X(:, 2), X * theta);
legend('Training data', 'Linear regression');
title('Linear Regression fit');
xlabel('sqft living');
ylabel('price');
hold off;

% Turkey-Anscombe plot
figure;
subplot(2, 1, 1);
residuals = y - X * theta; % residuals = actual values - predictions
plot(1:length(residuals), residuals, '.r', 'MarkerSize', 0.5);
hold on;
plot(1:size(X, 1), zeros(size(X, 1), 1)+1, '-g', 'LineWidth', 2);
legend('Difference to prediction', '0 line');
title('Turkey-Anscombe plot');
xlabel('adjusted values');
ylabel('residuals');
hold off;

subplot(2, 1, 2);
histogram(residuals, 50);
title('Turkey-Anscombe histogram'); % TODO labels

