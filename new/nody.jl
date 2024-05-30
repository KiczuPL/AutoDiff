abstract type GraphNode end
abstract type Operator <: GraphNode end


struct Constant{T <: Real} <: GraphNode
    output :: T
end

mutable struct ScalarOperator{F, T <: Real} <: Operator
    inputs :: Vector{GraphNode}
    output :: Vector{T}
    gradient :: Vector{T}
    name :: String
    ScalarOperator(fun, inputs::GraphNode...; name::String="?") = new{typeof(fun), T}(inputs, Vector{T}(), Vector{T}(), name)
end

mutable struct Variable{T <: Real} <: GraphNode
    output :: Matrix{T}
    gradient :: Matrix{T}
    name :: String
    Variable(output::Matrix{T}; name::String="?") = new{T}(output, zeros(eltype(output), length(output), length(output)), name)
end


mutable struct BroadcastedOperator{F, T <: Real} <: Operator
    inputs :: Vector{GraphNode}
    output :: Vector{T}
    gradient :: Vector{T}
    name :: String
    BroadcastedOperator(inputs::GraphNode...; name::String="?") = new{typeof(fun), T}(inputs, Vector{T}(), Vector{T}(), name)
end
