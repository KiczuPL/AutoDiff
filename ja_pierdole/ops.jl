# Funkcja sigmoidalna
sigmoid(x) = 1.0 / (1.0 + exp(-x))

using Random, Distributions

function xavier_init(input_dim::Int, output_dim::Union{Int,Nothing}=nothing)
    if output_dim === nothing
    scale = sqrt(2.0 / (input_dim + 1))
        return rand(Normal(0, scale), input_dim)
    end
    scale = sqrt(2.0 / (input_dim + output_dim))
    return rand(Normal(0, scale), input_dim, output_dim)
end

