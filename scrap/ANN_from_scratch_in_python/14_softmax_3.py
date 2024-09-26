import numpy as np

layer_outputs = [
    [4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]
]

print(f'sum of layer outputs:')
print(np.sum(layer_outputs, axis = 1 ))


print('---')
print(np.sum(layer_outputs, axis = 1 , keepdims=True))
print('----------------------------------------------')

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum( exp_values, axis = 1 , keepdims = True )


print(f'norm values:')
print(norm_values)

print('----------------------------------------------')

for i in norm_values:
    print(f'sum of norm values: {sum(i)}')



