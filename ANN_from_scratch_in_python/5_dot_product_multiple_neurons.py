import numpy as np
# 3 neuron with 3 inputs

#inputs won't change
inputs = [1, 2, 3, 2.5]

#1 set of weights for each neuron
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

#1 bias for each neuron
biases = [ 2, 3, 0.5]


output = np.dot(weights,inputs) + biases
print(output)




