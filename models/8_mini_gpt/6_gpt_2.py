import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

print('-------------------------------------------------------------------------')
print('Hyperparameters init')
#-------------------------------------------------------------------------
class HyperParameters():
    #-------------------------------------------------------------------------
    def __init__(self):

        self.batch_size = 16 # how many independent sequences will we process in parallel?
        self.block_size = 32 # what is the maximum context length for predictions?
        # self.max_iters = 5000
        self.max_iters = 10
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.device = self.get_device()
        self.eval_iters = 200
        self.n_embd = 64
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.0
        self.debug = False
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_device(self):
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

        return device
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
hyper = HyperParameters()

print('-------------------------------------------------------------------------')
print('Part 1 - read data')

#-------------------------------------------------------------------------
class SourceData():
    #-------------------------------------------------------------------------
    def __init__(self, file = 'input.txt', param_debug = False):
        self.file = file
        if param_debug:
            self.file = './models/8_mini_gpt/input.txt'
        
        with open(file = self.file, mode = 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))

        self.vocab_size = len(self.chars)

        # create a mapping from characters to integers
        self.str_to_idx = { char: idx  for idx,char in enumerate(self.chars) }
        self.idx_to_str = { idx : char for idx,char in enumerate(self.chars) }        
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def show_summary(self):
        print(f'here are all the unique characters that occur in this text:\n  {self.chars}')
        print(f'vocab size: {self.vocab_size}')

        print(f'first 5 character to integer mapping:')
        self.__sample_dict(self.str_to_idx)
        print(f'first 5 integer to character mapping:')
        self.__sample_dict(self.idx_to_str)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __sample_dict(self, dict, num_samples=5, print_result = True):
        
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
    def encode(self, str_input):
        return [ 
            self.str_to_idx[char] 
            for char in str_input
        ] # encoder: take a string, output a list of integers   
    #------------------------------------------------------------------------- 
    def decode(self, int_list):
        return ''.join(
            [
                self.idx_to_str[i] 
                for i in int_list
            ]
        ) # decoder: take a list of integers, output a string
    #------------------------------------------------------------------------- 
    #------------------------------------------------------------------------- 
    #-------------------------------------------------------------------------    
#-------------------------------------------------------------------------

src_d = SourceData(param_debug = hyper.debug)
src_d.show_summary()


print('-------------------------------------------------------------------------')
print('Part 3 - Train and test splits')

#-------------------------------------------------------------------------
class TrainValData():
    #-------------------------------------------------------------------------
    def __init__(self, text, encoder, percent_train = 0.9):
        data_selected = torch.tensor(encoder(text), dtype=torch.long) # encode the text into integers

        len_data_selected = len(data_selected)
        range_selected = int(percent_train*len_data_selected) # 0.9 * num = first 90% will be train, rest is val

        self.train_data = data_selected[:range_selected] # from 0 to range selected index (non inclusive)
        self.validation_data = data_selected[range_selected:] # from range selected to end of the list

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_batch(self, block_size = 32, batch_size = 16, device = 'cpu', str_split = 'train'):
        # generate a small batch of data of inputs x and targets y

        if str_split == 'train':
            data_selected = self.train_data
        else:
            data_selected = self.validation_data


        # this is cut short from the len minus the block size, if we select the last possible index, we need to be able
        #   to select block_size elements after it
        high_bound = len(data_selected) - block_size 
        size_selection = (batch_size,)        
     


        rand_int_tensor = torch.randint(high=high_bound, size=size_selection)

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
        x = torch.stack(
            [ 
                data_selected[i:i+block_size] 
                for i in rand_int_tensor
            ]
        )

        y = torch.stack(
            [
                data_selected[i+1:i+block_size+1] 
                for i in rand_int_tensor
            ]
        )

        x, y = x.to(device), y.to(device)
        
        return x, y        
        
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
train_val_data = TrainValData(text = src_d.text, encoder = src_d.encode)

#-------------------------------------------------------------------------
class EstimateLoss():
    #-------------------------------------------------------------------------
    # the operations performed within the decorated function will not be tracked for gradient computation
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(hyper.eval_iters)
            for k in range(hyper.eval_iters):
                X, Y = train_val_data.get_batch(str_split = split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Head(nn.Module):
    """ one head of self-attention """
    #-------------------------------------------------------------------------
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(hyper.n_embd, head_size, bias=False)
        self.query = nn.Linear(hyper.n_embd, head_size, bias=False)
        self.value = nn.Linear(hyper.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyper.block_size, hyper.block_size)))

        self.dropout = nn.Dropout(hyper.dropout)
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
        self.proj = nn.Linear(hyper.n_embd, hyper.n_embd)
        self.dropout = nn.Dropout(hyper.dropout)
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
            nn.Dropout(hyper.dropout),
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
        self.token_embedding_table = nn.Embedding(src_d.vocab_size, hyper.n_embd)
        self.position_embedding_table = nn.Embedding(hyper.block_size, hyper.n_embd)
        self.blocks = nn.Sequential(*[Block(hyper.n_embd, n_head=hyper.n_head) for _ in range(hyper.n_layer)])
        self.ln_f = nn.LayerNorm(hyper.n_embd) # final layer norm
        self.lm_head = nn.Linear(hyper.n_embd, src_d.vocab_size)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=hyper.device)) # (T,C)
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
            idx_cond = idx[:, -hyper.block_size:]
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
m = model.to(hyper.device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper.learning_rate)

for iter in range(hyper.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % hyper.eval_interval == 0 or iter == hyper.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = train_val_data.get_batch(str_split = 'train')

    

    # evaluate the loss
    logits, _loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    _loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hyper.device)
print(src_d.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
