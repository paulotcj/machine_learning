import torch

# torch.manual_seed(42)

# row_count, col_count = 4, 16

# input_vec : torch.Tensor = torch.rand((row_count, col_count))

input_vec = [
    [ 0.11  ,  0.22  , 0.33  , 0.44  ],
    [ 0.55  ,  0.77  , 0.22  , 0.11  ],
    [ 0.001 ,  0.002 , 0.999 , 0.003 ]
]

input_vec = torch.tensor(input_vec)


print(f'input_vec shape: {input_vec.shape}')
print(f'input_vec:\n{input_vec}')



print('----------------------------------------------')


# torch softmax as a reference
expected_softmax = torch.softmax(input_vec, dim=1)

print('\n\n')
print(f'expected_softmax:\n{expected_softmax}')


print('----------------------------------------------')
print('\n\n')




online_softmax = torch.zeros_like(input_vec)
print(f'online_softmax zeros:\n{online_softmax}')


print('----------------------------------------------')
print('\n\n')

print(f'input_vec.shape:{input_vec.shape}')
row_count = input_vec.shape[0]
col_count = input_vec.shape[1]
print(f'row_count: {row_count}')
print(f'col_count: {col_count}')
print('----------------------------------------------')
print('\n\n')


# we load the input vector in small blocks (adapted to the size of the `SRAM`) and compute 
# 2 statistics in a single pass:
# - the maximum value
# - the denominator

# The achievement lies in the fact that you are supposed to know the maximum value of the vector 
#   to compute the denominator.
# At each step, our knowledge of the maximum value may evolve (we may meet a value bigger than 
#   our precedent maximum).
# When it happens, we just adjust the result of our computation of the precedent step.


#-----------------------------------
for row_k, row_v in enumerate(input_vec):
    row_max = 0.0
    normalizer_term = 0.0
    print(f'row {row_k} -----------------------------------------------------')

    #-----------------------------------
    for col_k, col_v in enumerate(row_v):
        print(f'    col {col_k} ---------')

        #---- 
        old_row_max = row_max
        row_max = max(old_row_max, col_v)
        if old_row_max != row_max:
            print(f'        new max discovered: {row_max:.4f}')        
        #----

        # The adjustment procedure is based on rules of exponentiation: when multiplying a base raised 
        #   to one exponent by the same base raised to another exponent, the exponents add.
        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(col_v - row_max)

        print(f'        current row max: {row_max:.4f}, denominator: {normalizer_term:.4f}')
    #-----------------------------------

    #------
    # this section is pretty standard, you can compare with regular/safe softmax
    #   the only different thing is how 'normalizer_term' was calculated
    input_safe = input_vec[row_k] - row_max
    temp = torch.exp( input_safe ) / normalizer_term

    online_softmax[row_k] = temp
    #------

#-----------------------------------


print('----------------------------------------------')
print('\n\n')

print(f'torch.allclose(online_softmax, expected_softmax): {torch.allclose(online_softmax, expected_softmax)}')