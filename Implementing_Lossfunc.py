import numpy as np


sm_output = np.array([[0.7,0.1,0.2],
                      [0.1,0.5,0.4],
                      [0.02,0.9,0.08]])

class_targets = [0,1,1]

print(sm_output[range(len(sm_output)),class_targets])


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

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs , self.weights) + self.biases

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X,y = spiral_data(100,3)

class Activation_SoftMax():
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilties = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilties

class Activation_ReLU:
    def forward(self , inputs):
        self.output = np.maximum(0,inputs)


# Defining the Neural Networks
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_func = Loss_CategoricalCrossentropy()
loss = loss_func.calculate(activation2.output,y)
print("Loss : ",loss )