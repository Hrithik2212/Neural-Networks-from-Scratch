# Categorical Cross Entroy 

'''
Sample Loss = - (Target Value[i] * Log(Predicted Value[i]))
    where i is the sample index 
'''
import numpy as np
b = 5.2 
print(np.log(b))


import math

softmax_output = [0.7,0.1,0.2]
# calculate the loss on this output from final layer
target_output = [1,0,0]
target_class = 0 
loss = 0
for i,j in zip(softmax_output,target_output):
    loss += -(j*math.log(i))
print(loss)