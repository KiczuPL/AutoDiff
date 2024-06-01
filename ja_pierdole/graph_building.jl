include("nody.jl")

function visit(node::Node, visited::Set{Node}, order::Vector{Node})
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::OperationNode, visited::Set{Node}, order::Vector{Node})
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::Node)
    visited = Set{Node}()
    order = Vector{Node}()
    visit(head, visited, order)
    return order
end


function init_nodes!(order::Vector{Node})
    for node in order
        init_node!(node)
    end
end


init_node!(node::ConstantNode) = nothing
init_node!(node::InputNode) = nothing
init_node!(node::VariableNode) = nothing
function init_node!(node::OperationNode)
    output_size = size(forward(node, [input.output for input in node.inputs]...))
    node.output = zeros(output_size)
    node.gradient = zeros(output_size)
end