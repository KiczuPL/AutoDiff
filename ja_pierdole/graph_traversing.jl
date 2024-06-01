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


update!(node::ConstantNode, gradient) = nothing
update!(node::VariableNode, gradient) = node.gradient .+= gradient
update!(node::InputNode, gradient) = nothing
update!(node::Node, gradient) = node.gradient .+= gradient

function backward!(order::Vector{Node}; seed=1.0)
    result = last(order)
    result.gradient = [seed]
    # @assert length(result.output) == [1] "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::ConstantNode) end
function backward!(node::VariableNode) end
function backward!(node::InputNode) end
function backward!(node::OperationNode)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

function predict!(input::AbstractVecOrMat, input_node::InputNode, output_node::Node, order::Vector{Node})
    input_node.output .= input
    for node in order
        compute!(node)
    end
    return output_node.output
end



function predict!(order::Vector{Node})
    for node in order
        compute!(node)
    end
    return last(order).output
end