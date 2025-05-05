import torch
import numpy as np

# 预计返回W*H*64*63
def position_decoding(x, L):
    out = [x]
    for i in range(L):
        out.append(torch.sin(2 ** i * np.pi * x))
        out.append(torch.cos(2 ** i * np.pi * x))
    return torch.cat(out, dim=-1)