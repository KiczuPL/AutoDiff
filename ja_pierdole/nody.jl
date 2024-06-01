abstract type Node end

# abstract type Operator <: Node end

# Wejście x, dla niego nie liczymy gradientu (bo po co? xD) 
mutable struct InputNode <: Node
    output::AbstractVecOrMat
    name::String

    InputNode(output::AbstractVecOrMat, name="?"::String) = new(output, name)
end

struct ConstantNode <: Node
    output::AbstractVecOrMat
    ConstantNode(output) = new([output])
end

# Zmienna (np jakaś macierz wag), to właśnie jej wagami kręcimy, żeby optyamlizować sieć
mutable struct VariableNode <: Node
    output::AbstractVecOrMat
    gradient::AbstractVecOrMat
    name::String

    VariableNode(output::AbstractVecOrMat, name="?"::String) = new(output, zeros(size(output)), name)
end

# Operacja, dla niej liczymy gradient normalnie
mutable struct OperationNode{F} <: Node
    inputs::Vector{Node}
    output::Union{AbstractVecOrMat,Nothing}
    gradient::Union{AbstractVecOrMat,Nothing}

    OperationNode(fun::F, inputs::Vector{Node}) where {F} =
        new{F}(inputs, nothing, nothing)

end