include("graph.jl")

x = Variable(5.0, name="x")
y = Variable(2.0, name="y")
two = Constant(2.0)
squared = x^two
ysquared = y^two
sine = sin(squared)
ycos = cos(ysquared)
summed = sine + ycos
order = topological_sort(sine)


# y = forward!(order)
# println(y)
yb = backward!(order)
println(yb)