import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
#-------------------------------------------------------------------------
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
# max_iters = 3000
max_iters = 10_000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
str_to_idx = { # string to index 
                char:idx 
                for idx,char in enumerate(chars) 
             } 

idx_to_str = { # index to string
                idx:char 
                for idx,char in enumerate(chars) 
             } 
encode = lambda str: [str_to_idx[c]  # encoder: take a string, output a list of integers
                      for c in str] 

decode = lambda int_list: ''.join( # decoder: take a list of integers, output a string
                                [ idx_to_str[i] 
                                  for i in int_list ]
                              )
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
# super simple bigram model
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
            logits, loss = self(idx = idx) # get the predictions - the problem here is that we are transforming the embedding multiple times

            # focus only on the last time step, select all rows, but only the last column, and all channels
            logits = logits[:, -1, :] # from (B,T,C) becomes (B, C)
            
            # apply softmax to get probabilities - dim=-1 means we apply softmax to the last dimension
            probs = F.softmax(logits, dim=-1) # (B, C) -> (1, 65)
            
            # draw samples (1) from the distribution - the return is a index from the tensor
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
