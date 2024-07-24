import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
normalization_values = exp_values / np.sum(exp_values)
print('----------------------------------------------')


print(f'sum of normalization values: {sum(normalization_values)}')
print('normalization values:')
print(normalization_values)
