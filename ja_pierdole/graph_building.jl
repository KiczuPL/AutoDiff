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