import torch
import torch.nn as nn
from torch.nn import functional as F


#-------------------------------------------------------------------------
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
temp_ones = torch.ones(3, 3)
lower_triangular = torch.tril(input = temp_ones) # return a lower triangular part of the matrix

# computes the sum of elements along a specified dimension of a tensor
sum_lower_triangular = torch.sum(input = lower_triangular, dim = 1, keepdim=True) 

low_tri_div_sum_low_tri = lower_triangular / sum_lower_triangular

a = low_tri_div_sum_low_tri

b = torch.randint( low = 0, high = 10, size = (3,2) ).float() # size (3,2) - 3 rows, 2 columns
c = a @ b # matrix multiplication introduced in Python 3.5
print(f'temp_ones=\n{temp_ones}')
print(f'lower_triangular=\n{lower_triangular}')
print(f'sum_lower_triangular=\n{sum_lower_triangular}')
print(f'low_tri_div_sum_low_tri=\n{low_tri_div_sum_low_tri}')

print('------')
print(f'a=\n{a}')
print('--')
print(f'b=\n{b}')
print('--')
print(f'c=\n{c}')

print('-----------------------------------------------')

# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
print(f'x.shape={x.shape}')
print(f'x=\n{x}')

# We want x[b,t] = mean_{i<=t} x[b,i]
# The general idea here is that we want the tokens to talk with each other, but
#   only the preceding tokens, the future ones should not be accessible.
# And the way we enable 'talking' is rudimentary, by means of averaging of the preceding tokens.
x_bow = torch.zeros((B,T,C)) # bow = bag of words
#----------------------
for b in range(B):
    #--------
    for t in range(T):
        # one important part here is that the range selection at t+1 is because
        #   the range is exclusive. So in the first loop we would have t = 0 and
        #   the range would be 0:0, which is empty, and since we wanto to select at
        #   least one element, we need to add 1 to the range.
        x_prev = x[b,:t+1] # (t,C)

        # take the following example: 
        # tensor([[ 0.1808, -0.0700],
        #         [-0.3596, -0.9152]])
        # the mean would be: [ ((0.1808) + (-0.3596))/2 , ((-0.0700) + (-0.9152))/2 ] ->
        #   [ -0.1788 / 2 , -0.9852000000000001 / 2 ] -> [ -0.0894, -0.49260000000000004 ] 
        # and the result produced by the mean function is: tensor([-0.0894, -0.4926])
        x_mean = torch.mean(input = x_prev, dim = 0)
        x_bow[b,t] = x_mean
    #--------
#----------------------

print('-----------------------------------------------')
torch.manual_seed(42)
a = torch.ones(3,3)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print(f'a=\n{a}')
print('--')
print(f'b=\n{b}')
print('--')
print(f'c=\n{c}')
'''
We got 
a = tensor([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])

b = tensor([[2, 7],
            [6, 4],
            [6, 5]])

c = tensor([[14, 16],
            [14, 16],
            [14, 16]])

Shape of c=(rows of a,columns of b)=[3,2]


The operations are: 
    Dot product of Row 1 of a and Column 1 of b:
    c11 = (1*2)+(1*6)+(1*6) = 2+6+6 = 14

    Dot product of Row 1 of a and Column 2 of b:
    c12 = (1*7)+(1*4)+(1*5) = 7+4+5 = 16

    Dot product of Row 2 of a and Column 1 of b:
    c21 = (1*2)+(1*6)+(1*6) = 2+6+6 =  14

    Dot product of Row 2 of a and Column 2 of b:
    c22 = (1*7)+(1*4)+(1*5) = 7+4+5 = 16

    Dot product of Row 3 of a and Column 1 of b:
    c31 = (1*2)+(1*6)+(1*6) = 2+6+6 = 14

    Dot product of Row 3 of a and Column 2 of b:
    c32 = (1*7)+(1*4)+(1*5) = 7+4+5 = 16

What should be evident is that the first tensor is accessing all the elements of the array / sequence
and the second tensor is accessing the elements across sentences. So we talk with with the tokens/words
from the sentence being analyzed.
'''

# now have a look how it would look with a lower triangular matrix
torch.manual_seed(42)
a = torch.ones(3,3)
a = torch.tril(a)
b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print(f'a=\n{a}')
print('--')
print(f'b=\n{b}')
print('--')
print(f'c=\n{c}')
'''
Now if we look at the present example we have:
a = tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])

b = tensor([[2., 7.],
            [6., 4.],
            [6., 5.]])

c = tensor([[2., 7.],
            [8., 11.],
            [14., 16.]])

From here what we can see is that from the first row we have:
c11=(1*2)+(0*6)+(0*6)=2 and c12=(1*7)+(0*4)+(0*5)=7 -> [2 , 7]
which basically copies the first row of b into c. 
In other words, no other batches or sentences affect the first sentence.

Then we have: c21 =(1*2)+(1*6)+(0*6)=2+6=8 and c22=(1*7)+(1*4)+(0*5)=7+4=11 -> [8, 11]
As we can see the second from A [1., 1., 0.] allows the first and second rows of B to
communicate with each other. And the same principle applies to the 3rd row of A and B, and
any other tensor following the same pattern.

    See operations details below:
    Shape of c=(rows of a,columns of b)=[3,2]

    Row 1 of a: [1,0,0]
        Dot product with Column 1 of b: [2,6,6]
        c11=(1*2)+(0*6)+(0*6)=2

        Dot product with Column 2 of b: [7,4,5]
        c12 =(1*7)+(0*4)+(0*5)=7
        
        
    Row 2 of a: [1,1,0]
        Dot product with Column 1 of b: [2,6,6]
        c21 =(1*2)+(1*6)+(0*6)=2+6=8

        Dot product with Column 2 of b: [7,4,5]
        c22=(1*7)+(1*4)+(0*5)=7+4=11

    Row 3 of a: [1,1,1]
        Dot product with Column 1 of b: [2,6,6]
        c31=(1*2)+(1*6)+(1*6)=2+6+6=14

        Dot product with Column 2 of b: [7,4,5]
        c32=(1*7)+(1*4)+(1*5)=7+4+5=16    
'''

# now have a look how it would look with a lower triangular matrix and average
torch.manual_seed(42)
a = torch.ones(3,3)
a = torch.tril(a)
a = a / torch.sum(input = a, dim = 1, keepdim=True)

b = torch.randint(0, 10, (3,2)).float()
c = a @ b
print(f'a=\n{a}')
print('--')
print(f'b=\n{b}')
print('--')
print(f'c=\n{c}')

'''
Now if we look at the present example we have:
a = tensor([[1.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000],
            [0.3333, 0.3333, 0.3333]])
        
b = tensor([[2., 7.],
            [6., 4.],
            [6., 5.]])
            
c = tensor([[2.0000, 7.0000],
            [4.0000, 5.5000],
            [4.6667, 5.3333]])

Tensor A is averaging the values of tensor B. And as we saw previously, this can be used
as a form of communication between the tokens/words in the sentence.
But of course, averaging is a rudimentary form of communication, but the principle of how
to shape a tensor so the result of a matrix multiplication can be used to extract information
or features is demonstrated here.
'''

print('-----------------------------------------------')

# torch.manual_seed(1337)
'''
This is the wei after the averaging operation:
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
'''

B,T,C = 4,8,2 # batch, time, channels
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True) # this is the averaging operation

# wei is (T, T) and x is (B, T, C), but broadcasting takes care of the dimensions
#  wei will turn into (B, T, T) and x will reamin (B, T, C)
x_bow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)

comparison1 = torch.eq(x_bow, x_bow2)
print(f'Element-wise comparison: (x_bow and x_bow2)=\n{comparison1}')

comparison2 = torch.equal(x_bow, x_bow2)
print(f'Are the tensors exactly equal? (x_bow and x_bow2)=\n{comparison2}')

difference = x_bow - x_bow2
print(f'Element-wise differences:(x_bow and x_bow2)\n{difference}')

abs_difference = torch.abs(x_bow - x_bow2)
print(f'Absolute differences:(x_bow and x_bow2)\n{abs_difference}')

are_close = torch.allclose(x_bow, x_bow2, atol=1e-5)
print(f'Are the tensors approximately equal?:(x_bow and x_bow2)\n{are_close}')




print('-----------------------------------------------')


# Another example
# version 3
B,T,C = 4,8,2 # batch, time, channels
tril = torch.tril( torch.ones(T, T) )
wei_zeros = torch.zeros(T, T)
wei_mask = wei_zeros.masked_fill(mask = tril == 0, value = float('-inf')) # this tells us that the future tokens are not accessible
wei = F.softmax(wei_mask, dim = 1) # pay attention to the outcome of softmax, as this is the equivalent of the averaging operation
x_bow3 = wei @ x

are_close = torch.allclose(x_bow, x_bow3, atol=1e-5)
print(f'Are the tensors approximately equal?:(x_bow and x_bow3)\n{are_close}')

'''
tril = tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
               [1., 1., 0., 0., 0., 0., 0., 0.],
               [1., 1., 1., 0., 0., 0., 0., 0.],
               [1., 1., 1., 1., 0., 0., 0., 0.],
               [1., 1., 1., 1., 1., 0., 0., 0.],
               [1., 1., 1., 1., 1., 1., 0., 0.],
               [1., 1., 1., 1., 1., 1., 1., 0.],
               [1., 1., 1., 1., 1., 1., 1., 1.]])

wei_zeros = tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]])

wei_mask = tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                   [0.,   0., -inf, -inf, -inf, -inf, -inf, -inf],
                   [0.,   0.,   0., -inf, -inf, -inf, -inf, -inf],
                   [0.,   0.,   0.,   0., -inf, -inf, -inf, -inf],
                   [0.,   0.,   0.,   0.,   0., -inf, -inf, -inf],
                   [0.,   0.,   0.,   0.,   0.,   0., -inf, -inf],
                   [0.,   0.,   0.,   0.,   0.,   0.,   0., -inf],
                   [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]]) 

wei = tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
              [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
              [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
              [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
              [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])

If we were to pay attention to some detials, as in with the details of Softmax. 
Where Softmax(z)_i = exp(z_i) / sum(exp(z)) for all i
And: exp(0) = 1  , and: exp(-inf) = 0 (for all the -inf values)
    So for z = [0., -inf, -inf, -inf, -inf, -inf, -inf, -inf]
    exp(z) = [1, 0, 0, 0, 0, 0, 0, 0]
    sum(exp(z)) = 1 + 0 + 0 + 0 + 0 + 0 + 0 + 0 = 1
    1 / 1 = 1
    0 / 1 = 0 (for all the other values)
    Therefore, Softmax(z) = [1, 0, 0, 0, 0, 0, 0, 0]

Now the second line: [0., 0., -inf, -inf, -inf, -inf, -inf, -inf]
    We have the same principles for exp(0) = 1  , and: exp(-inf) = 0
    exp(z) = [1, 1, 0, 0, 0, 0, 0, 0]
    sum(exp(z)) = 1 + 1 + 0 + 0 + 0 + 0 + 0 + 0 = 2
    and 1 / 2 = 0.5 and 0 / 2 = 0
    Therefore, Therefore, Softmax(z) = [0.5, 0.5, 0, 0, 0, 0, 0, 0]

And the same principle applies to the other lines, where we can see the mechanisms
for creating an averaging operation.


Another interesting mechanism is the wei_zeros. These weights are initialized to zero, but
with time they are adjusted to better reflect the interactions strenght we want to have between
the tokens. Or how much of tokens from the past we want to know, to aggregate, to know about it.




'''

print('-----------------------------------------------')


'''
let's think of how the matrix multiplication works in regards to the self-attention mechanism.
suppose we have just 2 tokens representing 2 words, let's name them A and B. Each word is encoded with
a vector of 3 elements. So we have a 2x3 matrix.
Let's day A = [1, 2, 3] and B = [4, 5, 6], the matrix would be:
[[1, 2, 3],
 [4, 5, 6]]

Since it's a 2x3, one of the matrices need to be transposed, so we can have a 3x2 matrix.
[[1, 4],
 [2, 5],
 [3, 6]]

The result would be:
   A | 1 2 3 | AxA | AxB |
   B | 4 5 6 | BxA | BxB |
             | 1      4  |
             | 2      5  |
             | 3      6  |
               A      B

Notice we have AxA, AxB, BxA, BxB. AxA looks at the interaction between the tokens A and A, AxB looks at
the interaction of the tokan A with its next word B, BxA looks at the interaction of the token B with the
previous word A, and BxB looks at the interaction between the token B and B.
'''


# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)
print(f'x.shape: {x.shape}')
print(f'x[0][0]=\n{x[0][0]}')



# let's see a single Head perform self-attention
head_size = 16
key   = nn.Linear(C, head_size, bias=False) #(32, 16)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, 16) -> (4, 8, 16)
q = query(x) # (B, T, 16) -> (4, 8, 16)

k_transpose = k.transpose(-2, -1) # [4, 16, 8] -  switch the last two dimensions (dim at -2 takes place where dim at -1 was)
print('\n')
print(f'k           shape: {k.shape}')
print(f'k_transpose shape: {k_transpose.shape}') # [4, 16, 8]
print(f'\nk[0][0]=\n{k[0][0]}')
print(f'\nk_transpose[0][0]=\n{k_transpose[0][0]}')

# [4, 8, 8]
wei =  q @ k_transpose # (B, T, 16) @ (B, 16, T) ---> (B, T, T) | (4, 8, 16) @ (4, 16, 8) ---> (4, 8, 8)
print(f'\nwei.shape: {wei.shape}')
print(f'wei[0]=\n{wei[0]}')



mat_TT_ones = torch.ones(T, T) # [8, 8]
tril = torch.tril(mat_TT_ones) # [8, 8]
print(f'\nmat_TT_ones.shape: {mat_TT_ones.shape}')
print(f'mat_TT_ones=\n{mat_TT_ones}')

print(f'\ntril.shape: {tril.shape}')
print(f'tril=\n{tril}')



wei_masked = wei.masked_fill(tril == 0, float('-inf')) # [4, 8, 8]
wei_softmax = F.softmax(wei_masked, dim=-1) # [4, 8, 8])

print(f'\nwei_masked.shape: {wei_masked.shape}')
print(f'wei_masked[0]=\n{wei_masked[0]}')

print(f'\nwei_softmax.shape: {wei_softmax.shape}')
print(f'wei_softmax[0]=\n{wei_softmax[0]}')


v = value(x) # # (B, T, 16) -> [4, 8, 16]
print(f'\nv.shape: {v.shape}')
print(f'v[0]=\n{v[0]}')


# [...,m,n] @ [...,n,p] = [...,m,p] -> [4, 8, 8] @ [4, 8, 16] = [4, 8, 16]
out = wei_softmax @ v #[4, 8, 16]

print(f'\nout.shape: {out.shape}')
print(f'out[0]=\n{out[0]}')
exit()
