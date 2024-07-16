import numpy as np


inputs = [
    [  1.0, 2.0,  3.0,  2.5 ]
]

#1 set of weights for each neuron
weights = [
    [  0.2 ,  0.8 , -0.5,  1.0  ],
    [  0.5 , -0.91, 0.26, -0.5  ],
    [ -0.26, -0.27, 0.17,  0.87 ]
]

#1 bias for each neuron
biases = [ 2, 3, 0.5 ]


weights_transposed = np.array(weights).T
output = np.dot(inputs, weights_transposed) + biases

print(output)

print("\nLets break down what this means")
print('---')
print(f"Inputs: {inputs[0]}")
print(f'  Neuron 1 output: {output[0][0]} ')
print(f'  Neuron 2 output: {output[0][1]} ')
print(f'  Neuron 3 output: {output[0][2]} ')





