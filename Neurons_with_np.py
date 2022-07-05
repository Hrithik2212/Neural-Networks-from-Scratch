from audioop import bias
import numpy as np 


inputs = np.array([1.0, 2.0, 3.0, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([2.0, 3.0, 0.5])

def neuron(input,weights,bias=0):
    return np.dot(input,weights)+bias 

# print(neuron(inputs,weights,biases))


def layer_of_neurons(inp,weights,biass):
    return np.dot(inp,weights.T)+biases

print(layer_of_neurons(inputs,weights,biases))

