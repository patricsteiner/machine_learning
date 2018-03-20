%% 1) Load and visualize data
% NOTE: All double quotes in the csv have been removed beforehand! (why were they there in the first place?)
features = csvread('house_data.csv', 3, 2); % start from 3rd column (price) and 2nd row (skip header)
labels = features(:, 1);
features = features(:, 2:end);
file = fopen('house_data.csv');
feature_titles = split(replace(fgetl(file), '_', ' '), ',');
feature_titles(1:3) = []; % remove id, date, price labels
fclose(file);
figure('Name', 'Comparison of house features vs. price'); % display scatterplots for all features vs. the price
for i = 1:length(feature_titles)
    subplot(4, 5, i);
    scatter(features(:, i), labels, 0.5, 'r')
    ylabel('price');
    xlabel(feature_titles(i));
    set(gca,'FontSize', 8);
end
set(gcf, 'Position', [200, 200, 1000, 800]);

disp('press any key to continue...');
pause;

%% 1.5) Feature scaling (standardization) to avoid overflows and speed up gradient descent
for i = 1:size(features, 2) % first 3 columns are id, date and price, features start from column 4
    features(:, i) = (features(:, i) - mean(features(:, i))) / std(features(:, i));
end

%% 2) Linear regression for price using sqft_living
m = size(features, 1); % number of training samples
X = [ones(m, 1), features(:, 3)]; % feature matrix with an added column of 1s (column 3 is sqft_living)
y = labels;
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
plot(X(:, 2), residuals, '.r', 'MarkerSize', 0.5);
hold on;
plot(xlim(), [0, 0], '-g', 'LineWidth', 2);
legend('Difference to prediction', '0 line');
title('Turkey-Anscombe plot');
xlabel('prediction');
ylabel('residuals');
hold off;

subplot(2, 1, 2);
histogram(residuals, 50);
title('Turkey-Anscombe histogram');
xlabel('residuals');
ylabel('frequency');

