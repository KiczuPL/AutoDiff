include("nody.jl")
include("graph_building.jl")
include("graph_traversing.jl")
include("ops.jl")




import Base: +
Base.Broadcast.broadcasted(+, x::Node, y::Node) = BroadcastedOperator(+, [x, y])
forward(::OperationNode{typeof(+)}, x, y) = return x .+ y
backward(::OperationNode{typeof(+)}, x, y, g) = tuple(g, g)


import Base: -
Base.Broadcast.broadcasted(-, x::Node, y::Node) = OperationNode(-, [x, y])
forward(::OperationNode{typeof(-)}, x, y) = return x .- y
backward(::OperationNode{typeof(-)}, x, y, g) = tuple(g,-g)


import Base: *
import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::Node, x::Node) = OperationNode(mul!, [A, x])
forward(::OperationNode{typeof(mul!)}, A, x) = return A * x
backward(::OperationNode{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)


# x .* y (element-wise multiplication)
import Base: broadcast
broadcasted(*, x::Node, y::Node) = OperationNode(*, [x, y])
forward(::OperationNode{typeof(*)}, x, y) = return x .* y
backward(::OperationNode{typeof(*)}, x, y, g) = tuple(g .* y, g .* x)

# Base.Broadcast.broadcasted(*, x::Node, y::Node) = OperationNode(*, [x, y])
# forward(::OperationNode{typeof(*)}, x, y) = return x .* y
# backward(node::OperationNode{typeof(*)}, x, y, g) = let
#     𝟏 = ones(length(node.output))
#     Jx = diagm(y .* 𝟏)
#     Jy = diagm(x .* 𝟏)
#     tuple(Jx' * g, Jy' * g)
# end


import Base: sum
sum(x::Node) = OperationNode(sum, Node[x])
forward(::OperationNode{typeof(sum)}, x) = return [sum(x)]
backward(::OperationNode{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x.output))
    J = 𝟏'
    tuple(J' * g)
end


import Base: ^
^(x::Node, n::Node) = OperationNode(^, [x, n])
forward(::OperationNode{typeof(^)}, x, n) = return x.^n
backward(::OperationNode{typeof(^)}, x, n, g) = tuple(g .* n .* x .^ (n.-1), g .* log.(abs.(x)) .* x .^ n)


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
