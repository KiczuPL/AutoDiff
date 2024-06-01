abstract type Node end

# abstract type Operator <: Node end

# Wejście x, dla niego nie liczymy gradientu (bo po co? xD) 
mutable struct InputNode <: Node
    output::AbstractVecOrMat

    InputNode(output::AbstractVecOrMat) = new(output)
end

struct ConstantNode{T} <: Node
    output :: T
end

# Zmienna (np jakaś macierz wag), to właśnie jej wagami kręcimy, żeby optyamlizować sieć
mutable struct VariableNode <: Node
    output::AbstractVecOrMat
    gradient::AbstractVecOrMat
    name::String

    VariableNode(output::AbstractVecOrMat, name="?"::String) = new(output, zeros(Float64, size(output)), name)
end

# Operacja, dla niej liczymy gradient normalnie
mutable struct OperationNode{F} <: Node
    inputs::Vector{Node}
    output::AbstractVecOrMat
    gradient::AbstractVecOrMat

    OperationNode(fun::F, inputs::Vector{Node}, output_size::Tuple{Int, Int}) where F = 
        new{F}(inputs, zeros(output_size), zeros(output_size))

    OperationNode(fun::F, inputs::Vector{Node}, output_size::Tuple{Int}) where F = 
        new{F}(inputs, zeros(output_size), zeros(output_size))

end