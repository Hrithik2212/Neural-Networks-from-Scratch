from BackPropagation2 import *

class Activation_SoftMax():
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilties = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilties
    def backward(self,dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output , single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)        

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) ==2:
            y_true= np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true]-=1
        self.dinputs = self.dinputs / samples

# Implementing Sofmax derivative in Neural Networks

import numpy as np 
import nnfs

nnfs.init()

softmax_ouputs = np.array([[0.7,0.1,0.2],
                           [0.1,0.5,0.4],
                           [0.02,0.9,0.8]])

class_targets = np.array([0,1,1])
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_ouputs,class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_SoftMax()
activation.output = softmax_ouputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_ouputs,class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print("Gradients Combined Loss and Activation :")
print(dvalues1)
print('Gradients seprate loss and activation:')
print(dvalues2)

