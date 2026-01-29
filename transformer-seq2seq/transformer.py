import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from attention_masks import causal_mask


class TransformerSeq2Seq(nn.Module):
def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, layers, max_len):
super().__init__()
self.encoder = Encoder(src_vocab, d_model, n_heads, layers, max_len)
self.decoder = Decoder(tgt_vocab, d_model, n_heads, layers, max_len)


def forward(self, src, tgt):
memory = self.encoder(src)
mask = causal_mask(tgt.size(0)).to(tgt.device)
return self.decoder(tgt, memory, mask)