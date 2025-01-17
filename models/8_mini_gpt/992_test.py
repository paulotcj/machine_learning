import torch
import torch.nn as nn

# Define an embedding layer with 10 possible indices and embedding dimension of 3
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)

# Example input: a batch of indices
input_indices = torch.tensor([1, 2, 3, 4])

# Get the corresponding embeddings
output = embedding(input_indices)

print(output)