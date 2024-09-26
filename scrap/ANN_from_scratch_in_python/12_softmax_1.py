import math

layer_outputs = [4.8, 1.21, 2.385]
print('----------------------------------------------')


exp_values = []

for output in layer_outputs:
    exp_values.append( math.e ** output )


print(exp_values)

normalization_base = sum(exp_values)
normalization_values = []

for v in exp_values:
    normalization_values.append( v / normalization_base )
print('----------------------------------------------')
print(f'normalization base: {normalization_base}')
print(f'sum of normalization values: {sum(normalization_values)}')
print('\nnormalization values:')
print(normalization_values)