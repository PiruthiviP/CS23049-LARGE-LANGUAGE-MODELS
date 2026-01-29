import torch
import torch.nn as nn


class Decoder(nn.Module):
def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len):
super().__init__()
self.embed = nn.Embedding(vocab_size, d_model)
self.pos = nn.Embedding(max_len, d_model)
layer = nn.TransformerDecoderLayer(d_model, n_heads)
self.decoder = nn.TransformerDecoder(layer, num_layers)
self.fc = nn.Linear(d_model, vocab_size)


def forward(self, tgt, memory, tgt_mask):
seq_len, batch = tgt.shape
positions = torch.arange(seq_len).unsqueeze(1).to(tgt.device)
x = self.embed(tgt) + self.pos(positions)
out = self.decoder(x, memory, tgt_mask=tgt_mask)
return self.fc(out)