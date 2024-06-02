include("nody.jl")

function mean_squared_loss(y, ŷ)
  return ConstantNode(0.5) .* ((y .- ŷ) .^ ConstantNode(2))
end


# function build_net2(loss::Function, layers...)
#   input, output = build_layers(layers...)
#   println("input: ", input)
#   println("output: ", output)
#   println("DUUUPPAAA: ", size(layers[end].b.output))
#   graph = topological_sort(output)
#   net = Network(graph, find_all_variables(graph), input, output, output)
#   init_nodes!(net.graph)
#   return net
# end


function build_net(loss::Function, layers...)
  input, output = build_layers(layers...)
  println("input: ", input)
  println("output: ", output)
  desired_output = InputNode(size(layers[end].b.output))
  println("DUUUPPAAA: ", size(layers[end].b.output))
  loss_output = loss(output, desired_output)
  graph = topological_sort(loss_output)
  init_nodes!(graph)
  net = Network(graph, find_all_variables(graph), input, output,desired_output, loss_output)
  return net
end

function build_layers(layers...)
  first_layer = first(layers)
  x = InputNode(size(first_layer.Wx.output)[2])
  y = x
  for layer in layers
    # println("layer: ", layer)
    y = build_output(layer, y)
  end

  return x, y
end



mutable struct Network
  graph::Vector{Node}
  variables::Vector{VariableNode}
  input::InputNode
  output::Node
  desired_output::InputNode
  loss::Node
end

struct Dense{F}
    Wx::VariableNode
    b::Union{VariableNode,Nothing}
    activation::F
end

function Dense(dims::Pair{Int64, Int64}, activation; weight_init=randn) 
  println("new Dense: dims(Wx): ",(dims[2],dims[1]," b: ", dims[2]))
  Dense(VariableNode((dims[2],dims[1]), name="Wx"),VariableNode((dims[2]), name="b"), activation)
end



# function build_output(layer::Dense, input::Node)
#     return layer.activation(layer.Wx * input .+ layer.b)
# end

function build_output(layer::Dense, input::Node)
  result = layer.Wx * input
  if layer.b isa VariableNode
    result = result .+ layer.b
  end
  if layer.activation isa Function
    return layer.activation(result)
  end 
  return result
end



struct RNNCell{F,I,H,V,S}
    Wx::I
    Wh::H
    b::V
    activation::F
    state0::S
  end

  mutable struct Recurrent{T,S}
    cell::T
    state::S
  end
  

RNN(a...; ka...) = Recurrent(RNNCell(a...; ka...))
Recur(m::RNNCell) = Recurrent(m, m.state0)