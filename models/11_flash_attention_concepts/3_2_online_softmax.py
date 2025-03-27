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


# read, max output
row_max = []

# get the max tensor from each row
for tensor in input_vec:
    max_val = tensor[0]
    for i in tensor:
        max_val = max(max_val , i)
    row_max.append(max_val)

row_max = torch.tensor(row_max).unsqueeze(1) # convert it to tensor and then add 1 dim, from shape (4,) to (4, 1)
print(f'input row max\n{row_max}')

print('----------------------------------------------')
print('\n\n')


# make the iput safe (check safe softmax)
input_safe = input_vec - row_max

print(f'input_safe:\n{input_safe}')

print('----------------------------------------------')
print('\n\n')


softmax_numerator = torch.exp(input_safe)
print(f'softmax_numerator:\n{softmax_numerator}')

print('----------------------------------------------')
print('\n\n')


normalizer_term = torch.sum(softmax_numerator, dim=1)
print(f'normalizer_tem:\n{normalizer_term}')
print('----')
normalizer_term = normalizer_term.unsqueeze(1)
print(f'normalizer_tem:\n{normalizer_term}')



print('----------------------------------------------')
print('\n\n')


naive_softmax = softmax_numerator / normalizer_term

print(f'naive_softmax:\n{naive_softmax}')


print('----------------------------------------------')
print('\n\n')


# checks if all elements of tensors a and b are close to each other within a certain tolerance 
#   by default, it uses a relative tolerance of 1e-5 and an absolute tolerance of 1e-8
print(f'torch.allclose(naive_softmax, expected_softmax): {torch.allclose(naive_softmax, expected_softmax)}')


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


#-----------------------------------

for row_k, row_v in enumerate(input_vec):
    row_max = 0.0
    normalizer_term = 0.0
    print(f'row {row_k} -----------------------------------------------------')


    #-----------------------------------
    for col_k, col_v in enumerate(row_v):
        print(f'    col {col_k} ---------')
        val = col_v
        old_row_max = row_max
        row_max = max(old_row_max, val)

        normalizer_term = normalizer_term * torch.exp(old_row_max - row_max) + torch.exp(val - row_max)

        if old_row_max != row_max:
            print(f'        new max discovered: {row_max:.4f}')

        print(f'        current row max: {row_max:.4f}, denominator: {normalizer_term:.4f}')
    #-----------------------------------

    # leverage our 2 statistics
    online_softmax[row_k, :] = torch.exp(input_vec[row_k, :] - row_max) / normalizer_term
#-----------------------------------

exit()
print('----------------------------------------------')
print('\n\n')

print(f'torch.allclose(online_softmax, expected_softmax): {torch.allclose(online_softmax, expected_softmax)}')