import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Hyperparameters
# batch_size = 16
# block_size = 32
# learning_rate = 1e-3
# n_embd = 64
# n_head = 2
# n_layer = 4

# n_input = 64
# n_hidden = 100
# n_output = 3
# epochs_CLS = 15

class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, n_embd)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        # print(positions.size())
        return self.positional_embedding(positions)

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, mask):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)
        self.mask = mask

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        if self.mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        # attn_sum = torch.sum(wei, dim=-1)
        # print(attn_sum)
        return out, wei

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, mask):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, mask) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.1)
        self.num_heads = num_heads

    def forward(self, x):
        attn_maps = []
        out = []
        for h in self.heads:
            head_out, attn_map = h(x)
            out.append(head_out)
            attn_maps.append(attn_map)
        out = torch.cat(out, dim=-1)
        out = self.dropout(self.proj(out))
        # print("1:", len(attn_maps))
        # print("2:", len(attn_maps[0]))
        # print("3:", len(attn_maps[0][0]))
        # print("4:", len(attn_maps[0][0]))
        return out, attn_maps

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, mask):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, mask)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        sa_out, attn_maps = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, max_len, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = PositionalEncoding(n_embd, max_len)
        # self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, mask=False) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(x)
        x = token_embeddings + position_embeddings
        attn_maps_all = []
        for block in self.blocks:
            x, attn_maps = block(x)
            attn_maps_all.append(attn_maps)
        x = self.ln(x)
        return x.mean(dim=1), attn_maps_all

class FeedForwardClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, max_len, n_input, n_hidden, n_output, block_size):
        super().__init__()
        self.encoder = TransformerEncoderLayer(n_embd, n_head, n_layer, vocab_size, max_len, block_size)
        self.classifier = FeedForwardClassifier(n_input, n_hidden, n_output)

    def forward(self, x):
        x, attn_maps = self.encoder(x)
        x = self.classifier(x)
        return x, attn_maps
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, max_len, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = PositionalEncoding(n_embd, max_len)
        # self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, mask=True) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(x)
        x = token_embeddings + position_embeddings
        attn_maps_all = []
        for block in self.blocks:
            x, attn_maps = block(x)
            attn_maps_all.append(attn_maps)
        x = self.ln(x)
        return x, attn_maps_all

class Decoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, max_len, block_size):
        super().__init__()
        self.decoder = TransformerDecoderLayer(n_embd, n_head, n_layer, vocab_size, max_len, block_size)
        self.ffwd = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        x, attn_maps = self.decoder(x)
        logits = self.ffwd(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, attn_maps
