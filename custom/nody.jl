abstract type Node end

struct InputNode <: Node
    value::Matrix{Float64}
    grad::Matrix{Float64}
end

struct OperationNode <: Node
    op::Function
    inputs::Vector{Node}
    value::Matrix{Float64}
    grad::Matrix{Float64}
end

function InputNode(value::Matrix{Float64})
    InputNode(value, zeros(size(value)))
end

function OperationNode(op::Function, inputs::Vector{Node})
    OperationNode(op, inputs, zeros(size(inputs[1].value)), zeros(size(inputs[1].value)))
end
