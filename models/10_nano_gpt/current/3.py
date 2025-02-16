with open('input.txt', 'r') as f:
    text = f.read()

data = text[:1000] # first 1,000 characters
print(data[:100])


#-------------------------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(data)
# print(tokens[:24])


#-------------------------------------------------------------------------
import torch
buf = torch.tensor(tokens[:24 + 1])

print(f'buf:\n{buf}')
print(f'buf shape: {buf.shape}')
print(f'buf[:-1]:\n{buf[:-1]}')
print(f'buf[1:]:\n{buf[1:]}')
exit()

#-------------------------------------------------------------------------
x = buf[:-1].view(4, 6)
y = buf[1:].view(4, 6)

#-------------------------------------------------------------------------
print(x)
print(y)