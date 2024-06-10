include("nody.jl")

reset!(node::ConstantNode) = nothing
reset!(node::InputNode) = nothing
reset!(node::VariableNode) = fill!(node.gradient, zero(eltype(node.gradient)))
reset!(node::OperationNode) = fill!(node.gradient, zero(eltype(node.gradient)))
function reset!(order::Vector{Node})
    for node in order
        reset!(node)
    end
    return nothing
end


reset_operations!(node::ConstantNode) = nothing
reset_operations!(node::InputNode) = nothing
reset_operations!(node::VariableNode) = nothing
reset_operations!(node::OperationNode) = fill!(node.gradient, zero(eltype(node.gradient)))


compute!(node::ConstantNode) = nothing
compute!(node::InputNode) = nothing
compute!(node::VariableNode) = nothing
function compute!(node::OperationNode)
    # println("compute_node ", node)
    # println("input sizes: ", [size(input.output) for input in node.inputs])
    node.output = forward(node, [input.output for input in node.inputs]...)
    # println("compute_node-successful! ", node)
end

function forward!(order::Vector{Node})
    for node in order
        compute!(node)
        reset_operations!(node)
    end
    return last(order).output
end


update!(node::ConstantNode, gradient) = nothing
update!(node::VariableNode, gradient) = node.gradient .+= gradient
update!(node::InputNode, gradient) = nothing
update!(node::OperationNode, gradient) =
    let
        # println("update! ", typeof(node))
        # println("update! --size", size(node.gradient))
        # println("update! --sizeggg", size(gradient))
        node.gradient .+= gradient
        # node.gradient = clamp.(node.gradient, -25, 25)
    end

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
    # println("backward! ", typeof(node))
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        # if !(input isa ConstantNode)
        #     println("in_node: ", typeof(input))
        #     println("input: ", size(input.gradient))
        #     println("gradient: ", size(gradient))
        # end

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

function find_all_variables(order::Vector{Node})
    variables = VariableNode[]
    for node in order
        if node isa VariableNode
            push!(variables, node)
        end
    end
    return variables
end

function adjust!(variables::Vector{VariableNode}, lr)
    for variable in variables
        # println("variable: ", variable.name)
        # @show variable.output
        variable.output .-= lr .* variable.gradient
        # @show variable.output
    end
    return nothing
end