import torch
import torch.nn as nn
from torch.nn import functional as F

print('-------------------------------------------------------------------------')
print('Hyperparameters init')
debug = False

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
# max_iters = 5000
max_iters = 10
eval_interval = 100
learning_rate = 1e-3
# device = 'cpu'
# if torch.cuda.is_available():
#    device = 'cuda' 
#    print('using cuda acceleration')
# elif torch.backends.mps.is_built():
#     device = 'mps'
#     print('using mps acceleration')
# else:
#     device = 'cpu'
#     print('using cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

#-------------------------------------------------------------------------
def sample_dict(dict, num_samples=5, print_result = True):
    
    temp_list = []
    for idx, (key,val) in enumerate( dict.items() ):
        if idx >= num_samples:
            break
        str_temp = f'key:{key}, val:{val}\n' 
        temp_list.append( str_temp )

    if print_result:
        merged = ''.join(temp_list)
        print(merged)
    
    return temp_list
#-------------------------------------------------------------------------
print('-------------------------------------------------------------------------')
print('Part 1')
torch.manual_seed(1337)

file = 'input.txt'
if debug:
    file = './models/8_mini_gpt/input.txt'
with open(file = file, mode = 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
print(f'here are all the unique characters that occur in this text:\n  {chars}')


vocab_size = len(chars)
print(f'vocab size: {vocab_size}')

# create a mapping from characters to integers
str_to_idx = { char: idx  for idx,char in enumerate(chars) }
idx_to_str = { idx : char for idx,char in enumerate(chars) }
print(f'first 5 character to integer mapping:')
sample_dict(str_to_idx)
print(f'first 5 integer to character mapping:')
sample_dict(idx_to_str)
exit()

encode = lambda s: [str_to_idx[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([idx_to_str[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data_selected = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data_selected)) # first 90% will be train, rest val
train_data = data_selected[:n]
validation_data = data_selected[n:]

#-------------------------------------------------------------------------
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# the operations performed within the decorated function will not be tracked for gradient computation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Head(nn.Module):
    """ one head of self-attention """
    #-------------------------------------------------------------------------
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    #-------------------------------------------------------------------------
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    #-------------------------------------------------------------------------
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    #-------------------------------------------------------------------------
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# super simple bigram model
class BigramLanguageModel(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, _loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    _loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
