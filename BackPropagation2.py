import numpy as np

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 *np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        self.output = np.dot(inputs*self.weights)+self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)


class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
    
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] =0

class Loss:
    def calculate(self,output, y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return(data_loss)
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples