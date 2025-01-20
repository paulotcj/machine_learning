import torch
import torch.nn as nn
from torch.nn import functional as F

print('Hyperparameters init')
#-------------------------------------------------------------------------
class HyperParameters():
    #-------------------------------------------------------------------------
    def __init__(self):

        self.batch_size = 16 # how many independent sequences will we process in parallel?
        self.block_size = 32 # what is the maximum context length for predictions?
        # self.max_iters = 5000
        self.max_iters = 100
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.device = self.get_device()
        self.eval_iters = 200
        self.n_embd = 64
        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.0
        self.debug = True
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_device(self):
        device = 'cpu'
        if torch.cuda.is_available():
           device = 'cuda' 
           print('using cuda acceleration')
        elif torch.backends.mps.is_built():
            device = 'mps'
            print('using mps acceleration')
        else:
            device = 'cpu'
            print('using cpu')
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return device
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
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
print('Part 3 - Train and test splits')
#-------------------------------------------------------------------------
class TrainValData():
    #-------------------------------------------------------------------------
    def __init__(self, text, encoder, device = 'cpu', percent_train = 0.9):

        data_selected = torch.tensor(encoder(text), dtype=torch.long) # encode the text into integers

        len_data_selected = len(data_selected)
        range_selected = int(percent_train*len_data_selected) # 0.9 * num = first 90% will be train, rest is val

        self.train_data = data_selected[:range_selected] # from 0 to range selected index (non inclusive)
        self.validation_data = data_selected[range_selected:] # from range selected to end of the list
        self.device = device

    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_batch(self, block_size = 32, batch_size = 16, str_split = 'train'):
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

        x, y = x.to(self.device), y.to(self.device)
        
        return x, y        
        
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class EstimateLoss():
    #-------------------------------------------------------------------------
    def __init__(self, eval_iters, model, train_val_data, device):
        self.eval_iters = eval_iters
        self.model = model
        self.device = device
        self.train_val_data = train_val_data
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # the operations performed within the decorated function will not be tracked for gradient computation
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        #------
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)

            for k in range(self.eval_iters):

                X, Y = self.train_val_data.get_batch(str_split = split)

                logits, loss = self.model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()
        #------

        self.model.train()

        return out
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    #-------------------------------------------------------------------------
    # DONE
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head):
        super().__init__()
        #--------
        self.vocab_size = vocab_size   # 65
        self.n_embd     = n_embd       # 64
        self.block_size = block_size   # 32
        self.n_layer    = n_layer      # 4
        self.n_head     = n_head       # 4
        #--------

        block_list = [
            Block(n_embd = hyper.n_embd, n_head = self.n_head) 
            for _ in range(self.n_layer)
        ]

        #--------
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings = self.block_size, embedding_dim = self.n_embd)
        self.blocks          = nn.Sequential(*block_list) # sends the elements of the list as arguments
        self.layernorm_final = nn.LayerNorm(normalized_shape = self.n_embd) # final layer norm
        self.lang_model_head = nn.Linear(in_features = self.n_embd, out_features = self.vocab_size)
        #--------
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # DONE
    def forward(self, inpt_int_tnsr, targets=None):
        #targets - [16,32]
        B, T = inpt_int_tnsr.shape # B (16): batch size, T(32): sequence length (if we are generating text batch might be always 1, and T will grow typically to 32)

        val_range = torch.arange(T, device=hyper.device) #[32] tensor. [ 0, 1, 2, ..., 31] - sometimes T starts at 1 and it will grow after each call up to typically 32

        # idx and targets are both (B,T) tensor of integers
        #   remember these embeddings will change as the model trains
        token_embeddings    = self.token_embedding_table(inpt_int_tnsr) # (B,T,C) - [16, 32, 64] - sometimes [1,1,64], [1,2,64] ... [1,32,64]
        position_embeddings = self.position_embedding_table(val_range) # (T,C) - [32, 64] - sometimes [1,64], [2,64] ... [32,64]

        

        tkn_emb_plus_pos_emb = token_embeddings + position_embeddings # (B,T,C) - [16, 32, 64] - sometimes [1,1,64], [1,2,64] ... [1,32,64]
        x_blocks             = self.blocks(tkn_emb_plus_pos_emb) # (B,T,C) - [16, 32, 64] - sometimes [1,1,64], [1,2,64] ... [1,32,64]
        x_norm               = self.layernorm_final(x_blocks) # (B,T,C) - [16, 32, 64] - it's complicated - but generally speaking we want the data to be evenly distributed to avoid problems such as vanishing and exploding gradients, also improves convergence and enhanced generalization - also the same applies to the shape as the vars above

        logits = self.lang_model_head(x_norm) # (B,T,vocab_size) - [16, 32, 65]) - pass through a linear layer to get the logits (with weights and biases)

        if targets is None: #if we are just generating text, or in other words, if we are not training
            loss = None
        else: # training...
            ''' note: here we compare the chars generated from the model to the actual chars in the text,
              therefore, the last dimensin of the vocab size is not particularly important when calculating
              the cross entropy
            '''
            B, T, C = logits.shape # the C is new, typically 65
            logits = logits.view(B*T, C) # (B*T, C) -> (16*32, 65) -> (512, 65)
            targets = targets.view(B*T) # from [16,32] to [512]
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # DONE
    def generate(self, inpt_int_tnsr, max_new_tokens):
        # inpt_int_tnsr - [1,1] - tensor([[0]])

        # note: typically we see notations of [B,...] meaning the batch, and we defined the batch as, say 32 in this example
        #   but we keep seeing batch being represented as 1 here. The thing is, we are doing 1 batch only, so 1


        for i in range(max_new_tokens):

            if (i> 0) and (i%300 == 0 or i == max_new_tokens - 1):
                print(f'i {i} - inpt_int_tnsr.shape: {inpt_int_tnsr.shape} - inpt_sliced.shape: {inpt_sliced.shape}')
                print(f'        logits.shape: {logits.shape} - probs.shape: {probs.shape} - idx_next.shape: {idx_next.shape}')
                print('-------------')

            # crop idx to the last block_size tokens - [1,1], [1,2]  ... until [1,32] which will be its final shape until the end
            #   note that inpt_int_tnsr will also grow as initially is [1,1]
            inpt_sliced = inpt_int_tnsr[:, -self.block_size:] # block_size = 32. select all rows, but only the last 32 columns

            # get the predictions - we don't care about loss at this point. we are not training
            logits, loss = self(inpt_int_tnsr = inpt_sliced, targets = None) # [1,1,65] forward method. we are not training, so no targerts
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) - [1, 65]
            
            # apply softmax to get probabilities - from the previous step, we apply softmax to figure out which char to pick
            probs = F.softmax(logits, dim=-1) # (B, C) - [1,65]
            
            # sample from the distribution - pick one char. It's not guaranteed to be the the char with the highest probability value
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) - [1,1]
            
            # append sampled index to the running sequence
            inpt_int_tnsr = torch.cat((inpt_int_tnsr, idx_next), dim=1) # (B, T+1) - [1,2]


        return inpt_int_tnsr # [1, 2001]
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
########################################################################################
###########
########### PAUSED HERE
###########
########################################################################################
#-------------------------------------------------------------------------
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    #-------------------------------------------------------------------------
    # DONE
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # 64 // 4 = 16
        self.sa = MultiHeadAttention(num_heads = n_head, head_size = head_size)
        self.feed_forward = FeedFoward(n_embd = n_embd)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = n_embd)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = n_embd)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
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
# done
class GPTLike():
    #-------------------------------------------------------------------------
    # done
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, eval_iters, eval_interval,
                  max_iters, learning_rate, train_val_data, device):
        #-------
        self.vocab_size     = vocab_size
        self.n_embd         = n_embd
        self.block_size     = block_size
        self.n_layer        = n_layer
        self.n_head         = n_head
        self.eval_iters     = eval_iters
        self.eval_interval  = eval_interval
        self.max_iters      = max_iters
        self.device         = device
        self.learning_rate  = learning_rate
        self.train_val_data = train_val_data
        #-------

        self.model = BigramLanguageModel(
            vocab_size  = self.vocab_size, 
            n_embd      = self.n_embd, 
            block_size  = self.block_size, 
            n_layer     = self.n_layer, 
            n_head      = self.n_head
        )
        self.loss_obj = EstimateLoss(
            eval_iters      = self.eval_iters, 
            model           = self.model, 
            train_val_data  = self.train_val_data, 
            device          = self.device
        )

        self.m = self.model.to(hyper.device) # same object just a shortcut for the line below

        # create a PyTorch optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.__show_params_summary()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # done
    def __show_params_summary(self):
        # print the number of parameters in the model
        param_list = [
            p.numel() # number of elements in the tensor
            for p in self.m.parameters()
        ]
        param_sum = sum(param_list)
        param_sum /= 1e6
        print(f'{param_sum} M parameters')    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # done
    def train(self):

        for iter in range(self.max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1: # if we match a multiple of the milestones or if we are at the last iteration
                losses = self.loss_obj.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = train_val_data.get_batch(str_split = 'train')

            

            # evaluate the loss
            logits, _loss = self.model(inpt_int_tnsr = xb, targets = yb) #forward method
            self.optimizer.zero_grad(set_to_none=True)
            _loss.backward()
            self.optimizer.step()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    # done
    def generate(self):
        # generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device) # is zero intentional as this usually means a new line ('\n')?

        m_generate = self.m.generate(inpt_int_tnsr = context, max_new_tokens = 2000) #[1, 2001] - tensor([ 0, 13, 52,  ...], device='cuda:0')

        m_generate_list = m_generate[0].tolist() # [0, 13, 55, ... ]

        m_generate_list_decode = src_d.decode(int_list = m_generate_list) # not trained -> "\nAnd they brince?\n\nSTANLET:\nHe madest my be tongues..."


        print(m_generate_list_decode)        

    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

torch.manual_seed(1337)
hyper = HyperParameters()
src_d = SourceData(param_debug = hyper.debug)
src_d.show_summary()
train_val_data = TrainValData(text = src_d.text, encoder = src_d.encode, device = hyper.device)

gpt_like = GPTLike(
    vocab_size      = src_d.vocab_size, 
    n_embd          = hyper.n_embd, 
    block_size      = hyper.block_size, 
    n_layer         = hyper.n_layer, 
    n_head          = hyper.n_head,
    eval_iters      = hyper.eval_iters,
    eval_interval   = hyper.eval_interval,
    max_iters       = hyper.max_iters,
    learning_rate   = hyper.learning_rate,
    train_val_data  = train_val_data,
    device          = hyper.device
)

gpt_like.train()

gpt_like.generate()




