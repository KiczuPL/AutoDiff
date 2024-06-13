include("nody.jl")

function mean_squared_loss(y, ŷ)
  return ConstantNode(0.5) .* ((y .- ŷ) .^ ConstantNode(2))
end

function dummy_error(y, ŷ)
  return y .- ŷ
end

abstract type Layer end


function declare_net(loss::Function, layers...)
  desired_output = InputNode(size(layers[end].b.output)) # do podmienienia z b na Wx, bo jak nie będzie biasu w warstwie to się wywali  
  all_layers = collect(layers)
  net = Network([], all_layers, [], nothing, [], nothing, desired_output, nothing, loss)
  return net
end



function build_net(loss::Function, layers...)
  input, output = build_layers(layers...)
  desired_output = InputNode(size(layers[end].b.output)) # do podmienienia z b na Wx, bo jak nie będzie biasu w warstwie to się wywali  
  loss_output = loss(output, desired_output)
  graph = topological_sort(loss_output)
  init_nodes!(graph)
  all_layers = collect(layers)
  # push!(all_layers, ErrorLayer(loss_output, desired_output, output))
  net = Network(graph,all_layers, find_all_variables(graph), input, [input], output,desired_output, loss_output, loss)
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
  layers::Vector{Layer}
  variables::Vector{VariableNode}
  input::Union{InputNode,Nothing}
  input_sequences::Vector{InputNode}
  output::Union{Node,Nothing}
  desired_output::InputNode
  loss::Union{Node,Nothing}
  loss_function::Function
end

function unfold_net!(net::Network, n_sequences)
  for layer in net.layers
    if layer isa RNN
      layer.unfolded_t = n_sequences
    end
  end

  first_layer = first(net.layers)
  x = InputNode(size(first_layer.Wx.output)[2])
  y = x
  for layer in net.layers
      y = build_output(layer, y)
  end
  net.output = y
  y = net.loss_function(y, net.desired_output)
  net.loss = y
  graph = topological_sort(y)
  net.graph = graph
  all_inputs = filter(x -> x isa InputNode, graph)
  pop!(all_inputs) # ostatni to desired_output
  net.input_sequences = all_inputs
  net.variables = find_all_variables(graph)
  init_nodes!(graph)
end

function feed_with_sequence!(net::Network, sequences...)
   for (inputNode, value) in zip(net.input_sequences, sequences)
    inputNode.output = value
   end
end

function feed_desired_output!(net::Network, desired_output)
  net.desired_output.output = desired_output
end


struct Dense{F} <: Layer
    Wx::VariableNode
    b::Union{VariableNode,Nothing}
    activation::F
end

function Dense(dims::Pair{Int64, Int64}, activation; weight_init=randn) 
  println("new Dense: dims(Wx): ",(dims[2],dims[1]," b: ", dims[2]))
  Dense(VariableNode((dims[2],dims[1]), name="Wx"),VariableNode((dims[2],1), name="b"), activation)
end


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




mutable struct RNN{F,I,H,V,S} <: Layer
  Wx::I
  Wh::H
  b::V
  activation::F
  prev_state_otput::S
  unfolded_t::Int
end

function build_output(layer::RNN, input::Node)
  result = layer.Wx * input .+ layer.b
  result = layer.activation(result)
  for _ in 1:layer.unfolded_t
    result = build_next_rnn_layer(layer, result)
  end  
  return result
end

function build_next_rnn_layer(layer::RNN, prev_state::Node)
  next_input = VariableNode(size(layer.Wx.output)[2])
  result = ((layer.Wx * next_input) .+ (layer.Wh * prev_state)) .+ layer.b
  if layer.activation isa Function
    return layer.activation(result)
  end 
  return result
end

  function RNN(dims::Pair{Int64, Int64}, activation; weight_init=randn) 
    println("new RNN: dims(Wx): ",(dims[2],dims[1]," b: ", dims[2]))
    RNN(VariableNode((dims[2],dims[1]), name="Wxx"), VariableNode((dims[2],dims[2]), name="Whh"), VariableNode((dims[2],1), name="b"), activation, nothing, 0)
  end
  
  

  mutable struct ErrorLayer <: Layer
    err_output::Node
    desired_output::InputNode
    received_output::Node
    err_function::Function
  end