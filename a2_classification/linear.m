function l = linear(x, derivative)
%Linear linear activation function

if derivative
    l = ones(size(x));
else
    l = x;
end

end
