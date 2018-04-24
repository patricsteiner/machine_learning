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
        
        function obj = train(obj, X, y, alpha, n_iterations)
            %TRAIN trains the NN using backpropagation
            %you need to init the weights first!
            errorHistory = zeros(n_iterations, 1);
            for iter = 1:n_iterations
                obj = obj.forwardPropagate(X);
                errorHistory(iter) = mean((obj.predict(X) - y).^2);
                %backpropagate
                delta = obj.layers(end).activated - y;
                for i = length(obj.layers)-1:-1:1
                    nextLayer = obj.layers(i+1);
                    %calculate gradient
                    if nextLayer.hasBias
                        delta = delta(:, 2:end);
                    end
                    weightsGradient = 1/size(X, 1) * delta' * obj.layers(i).activated;
                    %update weights
                    obj.layers(i).weights = obj.layers(i).weights - alpha * weightsGradient;
                    %calculate delta
                    derivedValues = obj.layers(i).activation(obj.layers(i).values, true);
                    if (obj.layers(i).hasBias) 
                        derivedValues = [ones(size(derivedValues, 1), 1), derivedValues];
                    end
                    delta = delta * obj.layers(i).weights .* derivedValues;
                end
            end
            figure;
            plot(1:n_iterations, errorHistory);
            title('Neural Network Gradient Descent');
            xlabel('iteration');
            ylabel('error (MSE)');
        end
        
        function p = predict(obj, X)
        	obj = obj.forwardPropagate(X);
            p = obj.layers(end).activated;
        end
    end
end

