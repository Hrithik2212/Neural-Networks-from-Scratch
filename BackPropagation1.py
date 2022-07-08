def Neuron(inp,w,b=0):
    for i,j in zip(inp,w):
        out += i*j
        # print(out)
    out += b
    return out

import numpy as np

dvalues = np.array([[1,1,1],
                    [2,2,2],
                    [3,3,3]],dtype=np.float64)

weights = np.array([[0.2 , 0.8 , -0.5,1],
                    [0.5 , -0.91 , 0.26 , -0.5],
                    [-0.26,-0.27,0.17,0.87]]).T

dx = [sum(x)*dvalues[0] for x in weights ]

# dinputs is the gradient of the neuron function 
dinputs = np.array(dx)
# this dinput is only for a single gradient but 
# we have to calculate a batch of gradient 
dinputs = np.dot(dvalues,weights.T)

inputs = np.array([[1,2,3,2.5],
                   [2,5,-1,2],
                   [-1.5,2.7,3.3,-0.8]])

dweights = np.dot(inputs.T , dvalues)
print(dweights)

biases = np.array([[2,3,0.5]])

dbiases = np.sum(dvalues,axis=0,keepdims=True)

print(dbiases)


z = np.array([[1,2,-3,-4],
              [2,-7,-1,3],
              [-1,2,5,-1]],dtype=np.float64)

dvalues = np.array(range(1,13)).reshape((3,4))

drelu = np.zeros_like(z)

drelu[z>0] = 1
print(drelu)

drelu *= dvalues
print(drelu)

""" Implemeting Foward and Backward Pass together """

inputs = np.array([[1,2,3,2.5],
                   [2,5,-1,2],
                   [-1.5,2.7,3.3,-0.8]])
dvalues = np.array([[1,1,1],
                    [2,2,2],
                    [3,3,3]],dtype=np.float64)
weights = np.array([[0.2 , 0.8 , -0.5,1],
                    [0.5 , -0.91 , 0.26 , -0.5],
                    [-0.26,-0.27,0.17,0.87]]).T
biases = np.array([[2,3,0.5]])

# forward pass
layer_outputs = np.dot(inputs,weights)+biases
relu_outputs = np.maximum(0,layer_outputs)
print("Relu Output: \n",relu_outputs)

# backward pass
drelu = relu_outputs.copy()
drelu[layer_outputs<0]=0
print("Derivative of Relu: \n",drelu)

dinputs = np.dot(drelu ,weights.T)
dweights = np.dot(inputs.T,drelu)
dbiases = np.sum(drelu,axis=0,keepdims=True)

weights += -0.001 * dweights
biases += 0.001 * dbiases
print(weights)
print(biases)
