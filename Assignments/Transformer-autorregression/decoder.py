import torch
import torch.nn as nn
from attention_masks import generate_causal_mask

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt_embed = self.embedding(tgt)

        seq_len = tgt.size(1)
        causal_mask = generate_causal_mask(seq_len)

        attn_output, _ = self.attn(
            tgt_embed, tgt_embed, tgt_embed, attn_mask=causal_mask
        )

        return self.fc(attn_output)
