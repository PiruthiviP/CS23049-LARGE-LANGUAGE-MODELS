import torch
import torch.nn as nn
import torch.optim as optim
from transformer import TransformerSeq2Seq
pairs = [
    # --- Paraphrase ---
    ("AI improves healthcare", "AI enhances medical diagnosis and treatment"),
    ("Transformers process data in parallel", "Transformers handle sequences simultaneously"),
    ("Attention improves NLP accuracy", "Attention mechanisms increase NLP performance"),
    ("Machine learning helps", "Machine learning supports data driven decisions"),
    ("Deep learning models can", "Deep learning models can learn abstract features"),

    # --- Q&A ---
    ("What is self-attention", "Self attention relates each word to every other word"),
    ("Why is positional encoding required", "Positional encoding provides word order information"),
    ("What is autoregression", "Autoregression generates output tokens sequentially"),
    ("What is a transformer", "A transformer is a neural network using attention"),
    ("What is machine learning", "Machine learning enables systems to learn from data"),

    # --- Text generation ---
    ("In the future AI will", "In the future AI will automate decision systems"),
    ("Transformers are useful because", "Transformers are useful because they capture global context"),
    ("Neural networks are powerful because", "Neural networks are powerful because they learn complex patterns"),
    ("Data science helps organizations", "Data science helps organizations make better decisions"),
    ("Artificial intelligence is important", "Artificial intelligence is important for modern technology")
]


# ---------------- VOCAB ---------------- #
def tokenize(text): return text.lower().split()

vocab = set()
for s, t in pairs:
    vocab |= set(tokenize(s))
    vocab |= set(tokenize(t))

vocab = list(vocab)
word2idx = {w:i+2 for i,w in enumerate(vocab)}
word2idx["<pad>"] = 0
word2idx["<sos>"] = 1
idx2word = {i:w for w,i in word2idx.items()}

vocab_size = len(word2idx)

# ---------------- MODEL ---------------- #
model = TransformerSeq2Seq(vocab_size, embed_dim=64)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAIN ---------------- #
for epoch in range(300):
    total_loss = 0
    for src, tgt in pairs:
        src_ids = torch.tensor([[word2idx[w] for w in tokenize(src)]])
        tgt_tokens = ["<sos>"] + tokenize(tgt)
        tgt_ids = torch.tensor([[word2idx[w] for w in tgt_tokens]])

        outputs = model(src_ids, tgt_ids[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size),
                         tgt_ids[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.4f}")

torch.save(model.state_dict(), "seq2seq.pt")
