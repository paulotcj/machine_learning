import torch


matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('original matrix')
print(matrix)

matrix_t = matrix.transpose(-2,-1)
print('transposed matrix')
print(matrix_t)


#-------------------------




if torch.has_mps:
    print("MPS (Apple Silicon GPU) is available!")
else:
    print("MPS is not available.") 

device = torch.device("mps" if torch.has_mps else "cpu") 