inputs = [1,2,3,2.5]

weights1 = [2.1 ,3.4, 4.5 ,-1.2]
weights2 = [1.4 ,-2.1, 3, 1.7]
weights3 = [1.5 , -1 ,-0.6 , 2.3]

bias1 = 3
bias2 = 4
bias3 = 5

weights = [weights1,weights2,weights3]
biass = [bias1,bias2,bias3]

from pprint import pprint as pp
# pp(weights)
# Modelling three neurons 
from first_neuron import simple_neuron as single_neuron

def multiple_neurons(inp,ws,b):
    layer_out = [single_neuron(inp,x,y) for x,y in zip(ws,b)]
    return layer_out

print(multiple_neurons(inputs,weights,biass))
