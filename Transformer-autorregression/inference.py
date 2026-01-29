import torch
import torch.nn as nn

# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# MODEL (MUST MATCH TRAINING)
# =====================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        out, _ = self.attn(tgt, memory, memory)
        return self.fc(out)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim)
        self.decoder = Decoder(vocab_size, embed_dim, num_heads)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        return self.decoder(tgt, memory)

# =====================================================
# VOCAB (EXACTLY 28 TOKENS — MUST MATCH TRAINING)
# =====================================================

stoi = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "a": 3,  "b": 4,  "c": 5,  "d": 6,
    "e": 7,  "f": 8,  "g": 9,  "h": 10,
    "i": 11, "j": 12, "k": 13, "l": 14,
    "m": 15, "n": 16, "o": 17, "p": 18,
    "q": 19, "r": 20, "s": 21, "t": 22,
    "u": 23, "v": 24, "w": 25, "x": 26,
    "y": 27
}

itos = {v: k for k, v in stoi.items()}

VOCAB_SIZE = 92
EMBED_DIM = 64
NUM_HEADS = 4

BOS_ID = stoi["<bos>"]
EOS_ID = stoi["<eos>"]

# =====================================================
# LOAD MODEL
# =====================================================

model = TransformerSeq2Seq(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS
).to(device)

state_dict = torch.load("seq2seq.pt", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded successfully")

# =====================================================
# ENCODING / DECODING
# =====================================================

def encode(text):
    tokens = [stoi[c] for c in text.lower() if c in stoi]
    tokens.append(EOS_ID)
    return torch.tensor(tokens, dtype=torch.long)

def decode(tokens):
    chars = []
    for t in tokens:
        if t == EOS_ID:
            break
        if t > 2:
            chars.append(itos[t])
    return "".join(chars)

# =====================================================
# AUTOREGRESSIVE GENERATION
# =====================================================

def generate(model, src, max_len=50):
    src = src.unsqueeze(0).to(device)
    memory = model.encoder(src)

    ys = torch.tensor([[BOS_ID]], device=device)

    for _ in range(max_len):
        out = model.decoder(ys, memory)
        next_token = out[:, -1].argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == EOS_ID:
            break

    return ys[0]

# =====================================================
# RUN Q/A OR PARAPHRASE
# =====================================================

while True:
    text = input("\nEnter question / sentence (or 'exit'): ")
    if text.lower() == "exit":
        break

    src = encode(text)
    output_tokens = generate(model, src)

    print("Model output:", decode(output_tokens.tolist()))
