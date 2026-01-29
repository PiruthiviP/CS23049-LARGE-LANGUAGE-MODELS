import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim)
        self.decoder = Decoder(vocab_size, embed_dim)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        return self.decoder(tgt, memory)
