import torch


def causal_mask(size):
mask = torch.triu(torch.ones(size, size), diagonal=1)
return mask.masked_fill(mask == 1, float('-inf'))