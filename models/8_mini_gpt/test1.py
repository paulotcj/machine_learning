import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug_param', action='store_true', default=True, help='Enable debug mode')
parser.add_argument('--no-debug_param', action='store_false', dest='debug_param', help='Disable debug mode')
args = parser.parse_args()
debug = args.debug_param
print(f'debug: {debug}')

# To pass the parameter via the command line, run:
# python test1.py --debug_param  # to set debug to True
# python test1.py --no-debug_param  # to set debug to False
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset


# Dataset already downloaded
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it

file_path = 'input.txt'
if debug:
    # file_path = 'C://Users//paulo//source//repos//machine_learning//models//8_mini_gpt//input.txt'
    file_path = 'models//8_mini_gpt//input.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
    
print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'All chars:<start>{ ''.join(chars) }<end>' )
print(f'vocab_size: {vocab_size}')

# create a mapping from characters to integers
#-------------------------------------------------------------------------
str_to_idx = { # string to index 
                char:idx 
                for idx,char in enumerate(chars) 
             } 

idx_to_str = { # index to string
                idx:char 
                for idx,char in enumerate(chars) 
             } 
#-------------------------------------------------------------------------
encode = lambda str: [str_to_idx[c]  # encoder: take a string, output a list of integers
                      for c in str] 

decode = lambda int_list: ''.join( # decoder: take a list of integers, output a string
                                [ idx_to_str[i] 
                                  for i in int_list ]
                              )
#-------------------------------------------------------------------------

str_test = "hii there"
str_test_encoded = encode(str_test)
str_test_decoded = decode(str_test_encoded)
print(str_test_encoded)
print(str_test_decoded)



# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data_selected = torch.tensor(encode(text), dtype=torch.long)
print(data_selected.shape, data_selected.dtype)
print(data_selected[:1000]) # the 1000 characters we looked at earier will to the GPT look like this


# Let's now split up the data into train and validation sets
n = int( 0.9 * len(data_selected) ) # first 90% will be train, rest val
train_data = data_selected[:n] # from 0 to n-1
validation_data = data_selected[n:] # from n to end


block_size = 8
train_data[:block_size+1] # just testing


x = train_data[:block_size] # block_size = 8
y = train_data[1:block_size+1]

print(f'original: {train_data[:block_size+1]}')
print(f'       x: {x}')
print(f'       y:     {y}\n')

#-------------------------------------------------------------------------
for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f"when input is {context} the target: {target}")
#-------------------------------------------------------------------------



torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

#-------------------------------------------------------------------------
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data_selected = train_data if split == 'train' else validation_data

    #----------
    # ix = torch.randint(len(data_selected) - block_size, (batch_size,))

    # this is cut short from the len minus the block size, if we select the last possible index, we need to be able
    #   to select block_size elements after it
    high_bound = len(data_selected) - block_size 
    size_selection = (batch_size,)

    # note the high is exclusive
    rand_int_tensor = torch.randint(high=high_bound, size=size_selection) # produces a random integer tensor, where any value is in between 0 and high_bound
    
    # print(f'\nlen(data_selected): {len(data_selected)}, block_size: {block_size}, high_bound: {high_bound}')
    # print(f"size_selection: {size_selection}, ix:{rand_int_tensor}\n\n") # high_bound: 1003846, size_selection: (4,), ix:tensor([ 76049, 234249, 934904, 560986])
    #----------
    
    r'''
    there an concern if this is under list bounds, so let's test the limits.
    consider len(data_selected) = 100, block_size = 8
    then high_bound = 100 - 8 = 92, but since the high is exclusive, the last possible index is 91
    For the First list:
      if 'i' was 91, we would have: start idx = 91 (inclusive), end idx = 91 + 8 = 99 (exclusive)
      therefore we would get: 91, 92, 93, 94, 95, 96, 97, 98

    For the second list:
      if 'i' was 91, then we would have: start idx = i + 1 = 92 (inclusive), end idx = i + 1 + 8 = 91 + 1 + 8 = 100 (exclusive)
      therefore we would get: 92, 93, 94, 95, 96, 97, 98, 99
    '''
    # select a segment starting at index i (from random sampling indexes) with length of block_size
    x = torch.stack( [ data_selected[ i : i+block_size ] 
                      for i in rand_int_tensor 
                      ] 
                    )
    # select a segment virtually identical to x, but shifted by one position to the right
    y = torch.stack( [ data_selected[ i+1 : i+1+block_size ] 
                      for i in rand_int_tensor 
                      ] 
                    )
    return x, y
#-------------------------------------------------------------------------

xb, yb = get_batch('train')
print(f'inputs shape:{xb.shape}')
print(f'inputs:\n{xb}')
print(f'\ntargets shape:{ yb.shape}')
print(f'targets:\n{yb}')

print('----')

#-------------------------------------------------------------------------
for b in range(batch_size): # batch dimension
    for i in range(block_size): # time dimension
        context = xb[b, :i+1] # get all elements from row b, from column 0 to i+1 (non-inclusive) - also remember this is shifted by one behind
        target = yb[b,i] # get element at row b, column i - this is shifted by one ahead
        print(f"when input is {context.tolist()} the target: {target}")
#-------------------------------------------------------------------------


print(xb) # our input to the transformer


import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

#-------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(num_embeddings = vocab_size, embedding_dim = vocab_size)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, idx, targets=None):

        #note: Logit -> The raw predictions which come out of the last layer of the neural network

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) - typically batch size, block size/ time steps, vocab size/channels
        
        # #-------
        # print(f'idx shape: {idx.shape}\nidx:\n{idx}')
        # if targets is not None:
        #     print(f'targets shape: {targets.shape}\ntargets:\n{targets}')
        # print('---------------')
        # print(f'logits shape: {logits.shape}\nlogits: (they are the idx that went through token_embedding_table)\n{logits}')
        # #-------

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # batch size (4), time steps (8), channels (65)
            logits = logits.view(B*T, C) # we reorganize and keep the channels/ vocab size as the last and isolated dimension
            targets = targets.view(B*T) # target was not modified and has shape (4, 8)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # we provide the idx, which is transformed via the token_embedding_table (nn.Embedding) 
            logits, _loss = self(idx = idx) # get the predictions - the problem here is that we are transforming the embedding multiple times

            # focus only on the last time step, select all rows, but only the last column, and all channels
            logits = logits[:, -1, :] # from (B,T,C) becomes (B, C)
            
            # apply softmax to get probabilities - dim=-1 means we apply softmax to the last dimension
            probs = F.softmax(logits, dim=-1) # (B, C) -> (1, 65) - we could skip this step
            
            # draw samples (1) from the distribution - the return is a index from the tensor
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) - we could pick a random number between 0 and vocab_size
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
m = BigramLanguageModel(vocab_size = vocab_size) # vocab_size = 65
logits, loss = m(idx = xb, targets = yb)
print(f'logits shape (note 4 * 8 = 32): {logits.shape}\nlogits:\n{logits}')
print(f'loss:{loss}')

temp_torch_zeros = torch.zeros(size = (1, 1), dtype=torch.long) # one row one column: [[0]]
temp_m_generate = m.generate(
    idx             = temp_torch_zeros, 
    max_new_tokens  = 100
)
temp_m_generate_list = temp_m_generate[0].tolist()
temp_decode = decode( temp_m_generate_list )


# print(f'temp_torch_zeros: {temp_torch_zeros}')
# print(f'temp_m_generate:\n{temp_m_generate}')
# print(f'temp_m_generate_list:{temp_m_generate_list}')

print( temp_decode )




# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32
#-------------------------------------------------------------------------
# loop_steps = 10_000
loop_steps = 100
for steps in range(loop_steps): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train') # as usual 'yb' is shifted by one position to the right

    # evaluate the loss (and do the embeddings)
    logits, _loss = m(idx = xb, targets = yb)
    # print(f'    _loss: {_loss.item()}')

    # set gradients to None - this will have a lower memory footprint and slightly improve performance
    optimizer.zero_grad(set_to_none=True) 

    _loss.backward()
    optimizer.step()
#-------------------------------------------------------------------------

print(f'With Optimizer\nFinal Loss: {_loss.item()}')
exit()

temp_torch_zeros = torch.zeros((1, 1), dtype=torch.long)
temp_m_generate = m.generate(
    idx             = temp_torch_zeros, 
    max_new_tokens  = 500
)

temp_m_generate_list = temp_m_generate[0].tolist()

print(decode(temp_m_generate_list))


#-------------------------------------------------------------------------
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
temp_ones = torch.ones(3, 3)
a = torch.tril(temp_ones) # return a lower triangular part of the matrix

# computes the sum of elements along a specified dimension of a tensor
sum_a = torch.sum(input = a, dim = 1, keepdim=True) 

a = a / sum_a
b = torch.randint( low = 0, high = 10, size = (3,2) ).float() # size (3,2) - 3 rows, 2 columns
c = a @ b # matrix multiplication introduced in Python 3.5
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
#-------------------------------------------------------------------------