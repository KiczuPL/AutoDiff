include("nody.jl")
include("graph_building.jl")
include("graph_traversing.jl")




import Base: +
+(x::Node, y::Node) = OperationNode(+, [x, y],size(x.output))
# Base.Broadcast.broadcasted(+, x::Node, y::Node) = BroadcastedOperator(+, [x, y])
forward(::OperationNode{typeof(+)}, x, y) = return x .+ y
# backward(::OperationNode{typeof(+)}, x, y, g) = tuple(g, g)



# Base.Broadcast.broadcasted(-, x::Node, y::Node) = OperationNode(-, x, y)
# forward(::OperationNode{typeof(-)}, x, y) = return x .- y
# backward(::OperationNode{typeof(-)}, x, y, g) = tuple(g,-g)


import Base: *
import LinearAlgebra: mul!

# x * y (aka matrix multiplication)
*(A::Node, x::Node) = OperationNode(mul!, [A, x], (size(A.output)[1], last(size(x.output))))
forward(::OperationNode{typeof(mul!)}, A, x) = return A * x
# backward(::OperationNode{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)




# # x .* y (element-wise multiplication)
# Base.Broadcast.broadcasted(*, x::Node, y::Node) = OperationNode(*, [x, y])
# forward(::OperationNode{typeof(*)}, x, y) = return x .* y
# backward(node::OperationNode{typeof(*)}, x, y, g) = let
#     𝟏 = ones(length(node.output))
#     Jx = diagm(y .* 𝟏)
#     Jy = diagm(x .* 𝟏)
#     tuple(Jx' * g, Jy' * g)
# end




# import Base: sum
# sum(x::Node) = BroadcastedOperator(sum, x::Node)
# forward(::OperationNode{typeof(sum)}, x) = return sum(x)
# backward(::OperationNode{typeof(sum)}, x, g) = let
#     𝟏 = ones(length(x))
#     J = 𝟏'
#     tuple(J' * g)
# end

