import torch


#-------------------------------------------------------------------------
class LayerNorm1d: # (used to be BatchNorm1d)
  #-------------------------------------------------------------------------
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps # epsilon to avoid dividing by zero
    self.gamma = torch.ones(dim) # fill it up with ones
    self.beta = torch.zeros(dim) # fill it up with zeros
  #-------------------------------------------------------------------------
  #-------------------------------------------------------------------------
  def __call__(self, x): # x = torch tensor
    # calculate the forward pass
    xmean = x.mean(dim = 1, keepdim=True) # batch mean [32,1]
    xvar = x.var(dim = 1, keepdim=True) # batch variance [32,1]

    print('\n\n')
    # print(f'xmean shape: {xmean.shape}')
    # print(f'xmean:\n{xmean}')
    # print(f'xvar  shape: {xvar.shape}')
    # print(f'xvar:\n{xvar}')
    

    xvar_sqrt = torch.sqrt(xvar + self.eps) # [32,1]
    # print(f'xvar_sqrt shape: {xvar_sqrt.shape}')
    # print(f'xvar_sqrt:\n{xvar_sqrt}')
    

    x_minus_xmean = x - xmean # [32, 100]
    print(f'x_minus_xmean shape: {x_minus_xmean.shape}')
    print(f'x_minus_xmean:\n{x_minus_xmean}')
    exit()

    xhat = x_minus_xmean / xvar_sqrt # normalize to unit variance

    # self.gamma is a tensor filled with ones, self.beta is a tensor filled with zeros
    self.out = self.gamma * xhat + self.beta
    return self.out
  #-------------------------------------------------------------------------
  #-------------------------------------------------------------------------
  def parameters(self):
    return [self.gamma, self.beta]
  #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

dim = 100
torch.manual_seed(1337)
layernorm_module = LayerNorm1d(dim = dim)
x_rand_tensor = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors

print(f'x_rand_tensor type: {type(x_rand_tensor)} \nx_rand_tensor.shape: {x_rand_tensor.shape}')
print(f'x_rand_tensor:\n{x_rand_tensor}')



x_rand_tensor = layernorm_module(x_rand_tensor)
x_rand_tensor.shape