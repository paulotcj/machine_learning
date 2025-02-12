import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

#-------------------------------------------------------------------------
# Transformer Model
class SmallGPT(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, block_size=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))  # Positional Encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout),
            num_layers=n_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.block_size = block_size
        self.init_weights()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = self.embed(x) + self.pos_emb[:, :x.shape[1], :]
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------




import tiktoken

#-------------------------------------------------------------------------
class TextDataset(Dataset):
    #-------------------------------------------------------------------------
    def __init__(self, text, block_size):
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        self.tokens = self.encoding.encode(text)
        self.block_size = block_size
        self.vocab_size = self.encoding.n_vocab
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __len__(self):
        return len(self.tokens) // self.block_size
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)  # Shifted for next-token prediction
        return x, y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

from torch.amp import autocast, GradScaler

# Load dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

dataset = TextDataset(text, block_size=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Small batch size due to 10GB VRAM



# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = dataset.vocab_size

model = SmallGPT(vocab_size=vocab_size).to(device)


# Optimizer & Scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50_000)

scaler = GradScaler(device.type)  # Mixed precision



#-------------------------------------------------------------------------
# Training Loop
epochs = 1
gradient_accumulation_steps = 8  # To simulate larger batch size

for epoch in range(epochs):
    model.train()
    #---------------------
    total_loss = 0

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        with autocast(device.type):  # Mixed precision
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        # Gradient accumulation
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")
#-------------------------------------------------------------------------