include("nody.jl")

reset!(node::ConstantNode) = nothing
reset!(node::InputNode) = nothing
reset!(node::VariableNode) = fill!(node.gradient, 0)
reset!(node::OperationNode) = fill!(node.gradient, 0)

compute!(node::ConstantNode) = nothing
compute!(node::InputNode) = nothing
compute!(node::VariableNode) = nothing
compute!(node::OperationNode) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector{Node})
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end