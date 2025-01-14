import torch


matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('original matrix')
print(matrix)

matrix_t = matrix.transpose(-2,-1)
print('transposed matrix')
print(matrix_t)