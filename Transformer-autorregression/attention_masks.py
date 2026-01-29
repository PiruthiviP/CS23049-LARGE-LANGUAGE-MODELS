import torch

def generate_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
