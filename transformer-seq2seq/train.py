import torch
import torch.nn as nn
from transformer import TransformerSeq2Seq


model = TransformerSeq2Seq(1000, 1000, 512, 8, 6, 100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(10):
src = torch.randint(0, 1000, (10, 32))
tgt = torch.randint(0, 1000, (10, 32))


output = model(src, tgt[:-1])
loss = loss_fn(output.reshape(-1, 1000), tgt[1:].reshape(-1))


optimizer.zero_grad()
loss.backward()
optimizer.step()


print(f"Epoch {epoch}, Loss: {loss.item():.4f}")