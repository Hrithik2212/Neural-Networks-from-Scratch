import numpy as np
import nnfs
nnfs.init()

 
def step(x):
    if x>0:
        return 1
    else:
        return 0

def ReLU(x):
    if x>0:
        return x
    else: 
        return 0

def sigmoid(x):
    # Sigmoid has the vanishing gradient problem 
    return 1/(1+np.exp^(-x))

class Layer_Dense:
    activation_func = {'step':step,'ReLU':ReLU,"sigmoid":sigmoid}
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs , self.weights) + self.biases

class Activation_ReLU:
    def forward(self , inputs):
        self.output = ReLU(inputs)


X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0 ,2,-1,3.3 ,-2.7,1.1,2.2, -100 ]
outputs = list(map(ReLU,inputs))
print(outputs)