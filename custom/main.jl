include("graph_ops.jl")

function main()
    
# Parametry
input_dim = 3
hidden_dim = 5
output_dim = 2
seq_length = 10
learning_rate = 0.01


inputs = [InputNode(rand(input_dim, 1)) for _ in 1:seq_length]
targets = [rand(output_dim, 1) for _ in 1:seq_length]

cell = RNNCell(input_dim, hidden_dim, output_dim)

for epoch in 1:100
    println("Epoch: $epoch")
    outputs, states = rnn_forward(cell, inputs)
    println("straken")
    cell = rnn_backward(cell, inputs, states, outputs, targets, learning_rate)
end

# Wyświetlamy wyniki
println("Trained RNN output: ", outputs[end])
end

main()
