import torch
import torch.nn as nn


class Encoder(nn.Module):
def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len):
super().__init__()
self.embed = nn.Embedding(vocab_size, d_model)
self.pos = nn.Embedding(max_len, d_model)
layer = nn.TransformerEncoderLayer(d_model, n_heads)
self.encoder = nn.TransformerEncoder(layer, num_layers)


def forward(self, src):
seq_len, batch = src.shape
positions = torch.arange(seq_len).unsqueeze(1).to(src.device)
x = self.embed(src) + self.pos(positions)
return self.encoder(x)