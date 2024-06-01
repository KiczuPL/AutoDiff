include("nody.jl")
include("graph_building.jl")
include("graph_traversing.jl")




import Base: +
# Base.Broadcast.broadcasted(+, x::Node, y::Node) = BroadcastedOperator(+, [x, y], size(x.output))
+(x::Node, y::Node) = OperationNode(+, [x, y],size(x.output))
forward(::OperationNode{typeof(+)}, x, y) = return x .+ y
backward(::OperationNode{typeof(+)}, x, y, g) = tuple(g, g)


import Base: -
Base.Broadcast.broadcasted(-, x::Node, y::Node) = OperationNode(-, [x, y], size(x.output))
-(x::Node, y::Node) = OperationNode(-, [x, y], size(x.output))
forward(::OperationNode{typeof(-)}, x, y) = return x .- y
backward(::OperationNode{typeof(-)}, x, y, g) = tuple(g,-g)


import Base: *
import LinearAlgebra: mul!

# x * y (aka matrix multiplication)
*(A::Node, x::Node) = OperationNode(mul!, [A, x], (size(A.output)[1], last(size(x.output))))
forward(::OperationNode{typeof(mul!)}, A, x) = return A * x
backward(::OperationNode{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)



import Base: sum
sum(x::Node) = OperationNode(sum, [x], (1,))
forward(::OperationNode{typeof(sum)}, x) = return sum(x)
backward(::OperationNode{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x.output))
    J = 𝟏'
    tuple(J' * g)
end


import Base: ^
^(x::Node, n::Node) = OperationNode(^, [x, n], size(x.output))
forward(::OperationNode{typeof(^)}, x, n) = return x.^n
backward(::OperationNode{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

