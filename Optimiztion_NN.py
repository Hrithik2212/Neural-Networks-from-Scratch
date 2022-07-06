import nnfs
import matplotlib.pyplot as plt 
from nnfs.datasets import vertical_data ,spiral_data
from Implementing_Lossfunc import *
nnfs.init()

X, y = spiral_data(100,3)

# Defining the Neural Networks
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_SoftMax()

loss_func = Loss_CategoricalCrossentropy()

lowest_loss = 999999
best_d1_w = dense1.weights.copy()
best_d2_w = dense2.weights.copy()
best_d1_b = dense1.biases.copy()
best_d2_b = dense2.biases.copy()

for iterations in range(10000):
    dense1.weights += 0.02 * np.random.randn(2,3)
    dense1.biases += 0.02* np.random.rand(1,3)
    dense2.weights += 0.02 * np.random.randn(3,3)
    dense2.biases += 0.02 * np.random.rand(1,3)
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_func.calculate(activation2.output,y)

    predictions = np.argmax(activation2.output ,axis=1)
    accuracy = np.mean(predictions==y)
    if loss < lowest_loss:
        best_d1_w = dense1.weights.copy()
        best_d2_w = dense2.weights.copy()
        best_d1_b = dense1.biases.copy()
        best_d2_b = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_d1_w
        dense2.weights = best_d2_w
        dense1.biases = best_d1_b
        dense2.biases = best_d2_b

    print(f'Iteration: {iterations} loss:{loss}')
    print(f"Accuracy : {accuracy}")
