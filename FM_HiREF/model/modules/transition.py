import torch.nn as nn
import torch.nn.functional as F

from .swiglu import SwiGLU

class Transition(nn.Module):
    def __init__(self, dim, factor, use_ln=False):
        super().__init__()
        self.use_ln = use_ln

        # Multi Layer Perceptron
        hidden_chan = (dim * factor)
        if use_ln:
            self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, hidden_chan * 2, bias=False)
        self.swiglu = SwiGLU() 
        self.linear_out = nn.Linear(hidden_chan, dim, bias=False)

    def forward(self, x, mask=None):

        if self.use_ln:
            x = self.layer_norm(x)
        x = self.linear(x)
        x = self.swiglu(x) 
        x = self.linear_out(x)

        if mask is not None:
            x = x * mask[..., None]

        return x