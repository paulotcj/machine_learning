
def dot_product_array(inputs, weights) -> int:
    result : int = 0
    for n_input, weight in zip(inputs, weights):
        result += n_input * weight

    return result

# 1 neuron with 4 inputs

#inputs won't change
inputs = [1, 2, 3, 2.5]

#1 set of weights for each neuron
weights = [0.2, 0.8, -0.5 , 1.0]

#1 bias for each neuron
bias = 2

import numpy as np
output = np.dot(weights, inputs) + bias
print(output)

print('Local dot product')
output = dot_product_array(inputs, weights) + bias
print(output)



