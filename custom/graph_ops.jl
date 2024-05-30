include("nody.jl")

function add(x::Node, y::Node)
    return OperationNode((a, b) -> a .+ b, [x, y])
end

function mul(x::Node, y::Node)
    return OperationNode((a, b) -> a * b, [x, y])
end

function eval(node::InputNode)
    return node.value
end

function eval(node::OperationNode)
    input_values = map(eval, node.inputs)
    return node.op(input_values...)
end

struct RNNCell
    Wxh::Matrix{Float64}  # Wagi dla wejścia
    Whh::Matrix{Float64}  # Wagi dla stanu ukrytego
    Why::Matrix{Float64}  # Wagi dla wyjścia
    bh::Vector{Float64}   # Bias dla stanu ukrytego
    by::Vector{Float64}   # Bias dla wyjścia
end

function RNNCell(input_dim::Int, hidden_dim::Int, output_dim::Int)
    RNNCell(randn(hidden_dim, input_dim), randn(hidden_dim, hidden_dim),
            randn(output_dim, hidden_dim), randn(hidden_dim), randn(output_dim))
end

function rnn_forward(cell::RNNCell, inputs::Vector{InputNode})
    h = zeros(size(cell.Whh))  # Początkowy stan ukryty (kolumna)
    outputs = Vector{Matrix{Float64}}(undef, length(inputs))
    states = Vector{Matrix{Float64}}(undef, length(inputs))

    for t in 1:length(inputs)
        h = tanh.(cell.Wxh * inputs[t].value .+ cell.Whh * h .+ cell.bh)
        y = cell.Why * h .+ cell.by
        states[t] = h
        outputs[t] = y
    end

    return outputs, states
end

function rnn_backward(cell::RNNCell, inputs::Vector{InputNode}, states::Vector{Matrix{Float64}}, outputs::Vector{Matrix{Float64}}, targets::Vector{Matrix{Float64}}, learning_rate::Float64)
    dWxh, dWhh, dWhy = zeros(size(cell.Wxh)), zeros(size(cell.Whh)), zeros(size(cell.Why))
    dbh, dby = zeros(size(cell.bh)), zeros(size(cell.by))
    dh_next = zeros(size(states[1]))

    for t in length(inputs):-1:1
        dy = outputs[t] - targets[t]
        dWhy += dy * states[t]'
        dby += dy
        dh = (cell.Why' * dy) + dh_next
        dh_raw = (1 .- states[t].^2) .* dh  # Gradient through tanh
        dbh += dh_raw
        dWxh += dh_raw * inputs[t].value'
        dWhh += dh_raw * (t > 1 ? states[t-1] : zeros(size(states[1])))'
        dh_next = cell.Whh' * dh_raw
    end

    # Update weights
    cell.Wxh -= learning_rate * dWxh
    cell.Whh -= learning_rate * dWhh
    cell.Why -= learning_rate * dWhy
    cell.bh -= learning_rate * dbh
    cell.by -= learning_rate * dby

    return cell
end

