import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


#----------------------------------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #typically the weights are neurons x inputs, so for a 3 neuron layer with 4 inputs, the weights would be 4x3
        # but that requirest transposing the matrix, so we will go with the inputs x neurons format

        #note that we already have the weights in the right format, where previously we would need to transpose the matrix
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #returns an array of shape (n_inputs, n_neurons) with random values
        self.biases = np.zeros((1, n_neurons)) #fills up an array of shape (1, n_neurons) with zeros

    def forward(self, inputs):
        self.output = np.dot( inputs, self.weights ) + self.biases
#----------------------------------------------
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
#----------------------------------------------
class Activation_Softmax:
    def forward(self, inputs):
        overflow_prevention = inputs - np.max(inputs, axis = 1 , keepdims = True)
        exp_values = np.exp( overflow_prevention )
        probabilities = exp_values / np.sum( exp_values, axis = 1, keepdims = True )
        self.output = probabilities

#----------------------------------------------

X, y = spiral_data(samples=100, classes=3)
#----

#the inputs is 2 because of x, y coordinates. The number of neurons is mostly arbitrary (in this case)
dense1 = Layer_Dense(n_inputs=2, n_neurons=3) 
activation1 = Activation_ReLU()
#----

dense2 = Layer_Dense(n_inputs=3, n_neurons=3) #n_inputs = 3 since we had 3 neurons from the previous layer
activation2 = Activation_Softmax()
#----


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(f'outputs:')
print(activation2.output)











print('----------------------------------------------')














