import torch
import torch.nn as nn
from encoder import TransformerEncoder

vocab = {
    "[PAD]":0, "[MASK]":1,
    "Transformers":2, "use":3, "self":4, "attention":5,
    "Mars":6, "is":7, "called":8, "the":9, "red":10, "planet":11
}

inputs = torch.tensor([[2,3,1,5]])
targets = torch.tensor([[2,3,4,5]])

model = TransformerEncoder(len(vocab), 32)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    optimizer.zero_grad()
    output, _ = model(inputs)
    loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
    loss.backward()
    optimizer.step()

print("Training Completed")
