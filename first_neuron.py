
from pip import main


inputs = [1.2,5.1,2.1]
weights = [3.1,2.1,8.7]

bias = 3

output = 0

def simple_neuron(inp,w,b=0):
    '''
    Each Neuron has inputs and there are eqaul number of weights to the no of input and there  is 
    also bias for each neuron
    '''
    out=0
    try:
        for i,j in zip(inp,w):
            out += i*j
            # print(out)
        out += b
        return out
    except :
        if len(inp) != len(w):
            print("Error : Lenght of Inputs and Weights are to be the same")
           

print(simple_neuron(inputs,weights,bias))

