import torch
import torch.nn as nn
from masks import causal_mask

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(self.embed(x))


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt = self.embed(tgt)

        size = tgt.size(1)
        mask = causal_mask(size).to(tgt.device)

        tgt, _ = self.self_attn(tgt, tgt, tgt, attn_mask=mask)
        tgt, _ = self.cross_attn(tgt, memory, memory)

        return self.fc(tgt)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=64, heads=4):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model)
        self.decoder = Decoder(vocab_size, d_model, heads)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        return self.decoder(tgt, memory)
