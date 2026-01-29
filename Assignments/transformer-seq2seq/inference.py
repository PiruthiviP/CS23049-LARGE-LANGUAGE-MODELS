import torch
from transformer import TransformerSeq2Seq


model = TransformerSeq2Seq(1000, 1000, 512, 8, 6, 100)
model.eval()


src = torch.randint(0, 1000, (10, 1))
output = torch.tensor([[1]]) # <SOS>


for _ in range(15):
preds = model(src, output)
next_token = preds[-1].argmax(dim=-1).unsqueeze(0)
output = torch.cat([output, next_token], dim=0)


print(output)