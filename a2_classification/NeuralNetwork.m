classdef NeuralNetwork
    %NEURALNETWORK Represents a neural network, groups layers
    
    properties
        layers
    end
    
    methods
        function nn = NeuralNetwork(inputLayer, outputLayer)
            %NEURALNETWORK constructor, a NN needs at least input and
            %output layers
            nn.layers = [inputLayer, outputLayer];
        end
        
        function obj = addLayer(obj, layer)
            %ADDLAYER adds the layer before the output layer
            obj.layers = [obj.layers(1:end-1), layer, obj.layers(end)];
        end
        
        function obj = initWeights(obj, min, max)
            for i = 1:length(obj.layers)-1 % last layer has no weights
                size = obj.layers(i).size;
                if (obj.layers(i).hasBias)
                    size = size + 1;
                end
                obj.layers(i).weights = (max-min) .* rand(obj.layers(i+1).size, size) + min;
            end
        end
        
        function obj = forwardPropagate(obj, X)
            obj.layers(1).values = X;
            obj.layers(1).activated = obj.layers(1).activation(X, false);
            if obj.layers(1).hasBias
                obj.layers(1).activated = [ones(size(X, 1), 1), obj.layers(1).activated];
            end
            for i = 2:length(obj.layers)
                prevLayer = obj.layers(i-1);
                obj.layers(i).values = prevLayer.activated * prevLayer.weights';
                obj.layers(i).activated = obj.layers(i).activation(obj.layers(i).values, false);
                if obj.layers(i).hasBias
                    obj.layers(i).activated = [ones(size(obj.layers(i).values, 1), 1), obj.layers(i).activated];
                end
            end
        end
        
        function cost = cost(~, predicted, actual)
            cost = 1/size(predicted, 1) * sum(sum(-actual .* log(predicted) - (1-actual) .* log(1-predicted)));
        end
        
        function obj = train(obj, X, y, alpha, lambda, n_iterations, visualize)
            %TRAIN trains the NN using backpropagation
            %you need to init the weights first!
            m = size(X, 1);
            errorHistory = zeros(n_iterations, 1);
            for iter = 1:n_iterations
                obj = obj.forwardPropagate(X);
                %backpropagate
                delta = obj.layers(end).activated - y;
                cost = obj.cost(obj.layers(end).activated, y);
                for i = length(obj.layers)-1:-1:1
                    nextLayer = obj.layers(i+1);
                    %calculate gradient
                    if nextLayer.hasBias
                        delta = delta(:, 2:end);
                    end
                    weightsGradient = 1/m * delta' * obj.layers(i).activated;
                    %regularization (excluding bias)
                    cost = cost + lambda/(2*m) * sum(sum(obj.layers(i).weights(:, 2:end).^2)); 
                    weightsGradient(:, 2:end) = weightsGradient(:, 2:end) + lambda/m * obj.layers(i).weights(:, 2:end);
                    %update weights
                    obj.layers(i).weights = obj.layers(i).weights - alpha * weightsGradient;
                    %calculate delta
                    derivedValues = obj.layers(i).activation(obj.layers(i).values, true);
                    if (obj.layers(i).hasBias) 
                        derivedValues = [ones(size(derivedValues, 1), 1), derivedValues];
                    end
                    delta = delta * obj.layers(i).weights .* derivedValues;      
                end
                errorHistory(iter) = cost;
                if (mod(iter, 100) == 0)
                    disp(cost);
                end
            end
            if visualize
                figure;
                plot(1:n_iterations, errorHistory);
                title('Neural Network Gradient Descent');
                xlabel('iteration');
                ylabel('error (log loss)');
            end
        end
        
        function p = predict(obj, X, binary)
        	obj = obj.forwardPropagate(X);
            p = obj.layers(end).activated;
            if binary
                p = p >= .5;
            end
        end
    end
end

