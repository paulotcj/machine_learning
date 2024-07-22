import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() 
X, y = spiral_data(100, 3)   



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



layer1 = Layer_Dense(n_inputs=2,n_neurons=5)
activation1 = Activation_ReLU()

layer1.forward(X)
print('----------------------------------------------')
print(layer1.output)
print('----------------------------------------------')
activation1.forward(layer1.output)
print(activation1.output)
print('----------------------------------------------')



# # layer1 = Layer_Dense(n_inputs=4, n_neurons=5)
# layer2 = Layer_Dense(n_inputs=5, n_neurons=2)

# print('----------------------------------------------')
# print(f'layer1.weights: \n{layer1.weights}')
# print(f'layer2.weights: \n{layer2.weights}')
# print('----------------------------------------------')

# layer1.forward(X)
# # print(layer1.output)
# # exit()
# layer2.forward(layer1.output)
# print(layer2.output)









