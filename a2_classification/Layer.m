classdef Layer
    %LAYER Represents a layer in a neural network
    
    properties
        size
        hasBias
        values
        activated
        weights
        activation
    end
    
    methods
        function layer = Layer(size, hasBias, activation)
            %LAYER Constructor
            layer.size = size;
            layer.hasBias = hasBias;
            layer.activation = activation;
        end
    end
end

