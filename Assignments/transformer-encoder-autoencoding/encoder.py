import torch.nn as nn
from attention import SelfAttention
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        attn_out, weights = self.attention(x)
        x = self.ffn(attn_out)
        return self.output(x), weights
