import numpy as np

# Now the simulation is for a middle layer, with 3 neurons and 4 inputs
# that means these neurons will be connected to the 3 neurons in the previous layer

inputs = [
    [  1.0, 2.0,  3.0,  2.5 ],
    [  2.0, 5.0, -1.0,  2.0 ],
    [ -1.5, 2.7,  3.3, -0.8 ],
]

#1 set of weights for each neuron
weights = [
    [  0.2 ,  0.8 , -0.5,  1.0  ],
    [  0.5 , -0.91, 0.26, -0.5  ],
    [ -0.26, -0.27, 0.17,  0.87 ]
]

#1 bias for each neuron
biases = [ 2, 3, 0.5]

# A problem with this approach arises from performing the dot product of the weights and inputs
#  the format provided for the inputs is 3 neurons with 4 inputs (3x4)
#  and we have a hidden layer with 3 neurons and and 4 weights (3x4)
#  so the operation below will fail:
#    output = np.dot(weights,inputs) + biases

# And the definition of a matrix dot product (matrix multiplication): MxN * NxP = MxP
#  meaning, multiply the rows of the first matrix by the columns of the second matrix
#  so if you have a 2 row 3 columns (2r,3c) matrix and a 3 row 2 columns (3r, 2c) matrix
#  the result will be a 2 row 2 columns (2r,2c) matrix

# To achieve that we need to transpose something. And we will go with the weights of the hidden layer

inputs_transposed = np.array(inputs).T
output = np.dot(weights,inputs_transposed) + biases

print(output)




