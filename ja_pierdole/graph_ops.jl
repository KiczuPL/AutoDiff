include("nody.jl")
include("graph_building.jl")
include("graph_traversing.jl")
include("ops.jl")
include("loss.jl")
include("cells.jl")


import Base: +
+(x::Node, y::Node) = OperationNode(+, Node[x, y])
forward(::OperationNode{typeof(+)}, x, y) = return x .+ y
backward(::OperationNode{typeof(+)}, x, y, g) = tuple(g, g)


# import Base: +
# Base.Broadcast.broadcasted(+, x::Node, y::Node) = BroadcastedOperator(+, [x, y])
# forward(::OperationNode{typeof(+)}, x, y) = return x .+ y
# backward(::OperationNode{typeof(+)}, x, y, g) = tuple(g, g)


import Base: -
Base.Broadcast.broadcasted(-, x::Node, y::Node) = OperationNode(-, Node[x, y])
forward(::OperationNode{typeof(-)}, x, y) = return x .- y
backward(::OperationNode{typeof(-)}, x, y, g) = tuple(g, -g)


import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::Node, x::Node) = OperationNode(mul!, Node[A, x])
forward(::OperationNode{typeof(mul!)}, A, x) = return A * x
backward(::OperationNode{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)


# x .* y (element-wise multiplication)
import Base: broadcast
broadcasted(*, x::Node, y::Node) = OperationNode(*, Node[x, y])
forward(::OperationNode{typeof(*)}, x, y) = return x .* y
# backward(::OperationNode{typeof(*)}, x, y, g) = tuple(g .* y, g .* x)
# backward(::OperationNode{typeof(*)}, x, y, g) = tuple(g .* y, x .* g)
backward(node::OperationNode{typeof(*)}, x, y, g) =
    let
        return tuple(g .* y, g .* x)
    end

import Base: sum
sum(x::Node) = OperationNode(sum, Node[x])
forward(::OperationNode{typeof(sum)}, x) = return [sum(x)]
# backward(::OperationNode{typeof(sum)}, x, g) = tuple(g .* ones(size(x)))
# JEBIE SIE NA BACKWARDZIE
backward(::OperationNode{typeof(sum)}, x, g) =
    let
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * g)
    end


import Base: ^
^(x::Node, n::Node) = OperationNode(^, Node[x, n])
forward(::OperationNode{typeof(^)}, x, n) = return x .^ n
backward(::OperationNode{typeof(^)}, x, n, g) =
    let
        return tuple(g .* n .* x .^ (n .- 1), g .* log.(abs.(x)) .* x .^ n)
    end


# tanh function overload with forward and backward methods
import Base: tanh
tanh(x::Node) = OperationNode(tanh, Node[x])
forward(::OperationNode{typeof(tanh)}, x) = return tanh.(x)
backward(::OperationNode{typeof(tanh)}, x, g) = tuple(g .* (1 .- tanh.(x) .^ 2))

# sigmoid function overload with forward and backward methods
import Base: broadcast
sigmoid(x::Node) = OperationNode(sigmoid, Node[x])
forward(::OperationNode{typeof(sigmoid)}, x) = return sigmoid.(x)
backward(::OperationNode{typeof(sigmoid)}, x, g) = tuple(g .* sigmoid.(x) .* (1 .- sigmoid.(x)))



# cross_entropy_loss(y, ŷ) = OperationNode(cross_entropy_loss, Node[y, ŷ])
# forward(::OperationNode{typeof(cross_entropy_loss)}, y, ŷ) = return [-sum(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ))]
# backward(::OperationNode{typeof(cross_entropy_loss)}, y, ŷ, g) = tuple(-g .* y ./ ŷ .+ g .* (1 .- y) ./ (1 .- ŷ), g .* log.(ŷ) ./ ŷ .- g .* log.(1 .- ŷ) ./ (1 .- ŷ))




cross_entropy_loss(y_hat::Node, y::Node) = OperationNode(cross_entropy_loss, Node[y_hat, y])
forward(::OperationNode{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        # y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return [loss]
    end
backward(::OperationNode{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        # y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end

