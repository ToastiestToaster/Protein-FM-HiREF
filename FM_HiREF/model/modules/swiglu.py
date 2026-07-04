import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x