import torch

ones_matrix = torch.ones(6,6)

print(ones_matrix)


upper_triangular_mask = torch.triu(ones_matrix, diagonal=0)
print('\n\n')
print(upper_triangular_mask)


upper_triangular_mask = torch.triu(ones_matrix, diagonal=1)
print('\n\n')
print(upper_triangular_mask)


upper_triangular_mask = torch.triu(ones_matrix, diagonal=2)
print('\n\n')
print(upper_triangular_mask)

print('-------------------------')

ones_matrix = torch.ones(6,6,6)

print(ones_matrix)

upper_triangular_mask = torch.triu(ones_matrix, diagonal=2)
print('\n\n')
print(upper_triangular_mask)