
function mean_squared_loss(y, ŷ)
    return ConstantNode(0.5) .* ((y .- ŷ) .^ ConstantNode(2))
end