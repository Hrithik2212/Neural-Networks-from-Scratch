import nnfs
import numpy as np
from nnfs import datasets
nnfs.init()

class Layer_Dense:
    def __init__(self,inputs,n_neurons):
        self.weights = 0.01 *np.random.randn(inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)


class Activation_Relu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] =0

class Loss:
    def calculate(self,output, y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
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


class Activation_SoftMax():
    def forward(self,inputs):
        self.inputs = inputs
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

class Optimizer_SGD:
    def __init__(self,learning_rate=.85,decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate  = self.learning_rate*(1./(1.+self.decay*self.iterations)) 
    
    def update_params(self,layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    
    def post_update_params(self):
        self.iterations +=1

X , y = datasets.spiral_data(100,3)

# Initialising the Neural Network class
class DNN_2layers:
    def train(self,X,y,epochs,learning_rate=1,decay=1e-2):
        # Intialisation of the Layers 
        dense1 = Layer_Dense(2,64)
        activation1 = Activation_Relu()
        dense2 = Layer_Dense(64,3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        optimizer = Optimizer_SGD(learning_rate,decay)

        # Training
        for epoch in range(epochs):
            # Forward Pass
            dense1.forward(X)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            loss  = loss_activation.forward(dense2.output,y)

            predictions = np.argmax(loss_activation.output,axis=1)
            if len(y.shape) ==2:
                y= np.argmax(y,axis=1)
            accuracy = np.mean(predictions==y)

            if not epoch%100:
                print(f'Epoch : {epoch}, ')
                print(f'Accuracy : {accuracy}   Loss : {loss} ')
                print(f'Learning Rate : {optimizer.current_learning_rate}')
            # Backward Pass
            loss_activation.backward(loss_activation.output,y)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()
        return predictions

dnn = DNN_2layers().train(X,y,20001,2.8,1e-3)