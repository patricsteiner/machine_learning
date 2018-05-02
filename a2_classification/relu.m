function r = relu(x, derivative)
%SIGMOID Computes the relu function or its derivative if derivative = true

if derivative
    r = double(x > 0);
else
    r = max(0, x);
end

end
