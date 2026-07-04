import torch
import torch.nn as nn

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cond = nn.LayerNorm(dim_cond)

        # Gamma
        self.g_linear = nn.Linear(dim_cond, dim)

        # Beta
        self.b_linear = nn.Linear(dim_cond, dim, bias=False)

    def forward(self, x, cond, mask = None):
        normed_x = self.norm(x)
        normed_cond = self.norm_cond(cond)
        
        # Condition -> Gamma
        gamma = self.g_linear(normed_cond)
        gamma = torch.sigmoid(gamma)

        # Condition -> Beta
        beta = self.b_linear(normed_cond)

        # x * gamma (scale) + beta (shift)
        out = normed_x * gamma + beta
        if mask is not None:
            out = out * mask[..., None]
        
        return out

class AdaptiveOutputScale(nn.Module):
    def __init__(self, dim, dim_cond, bias_init = -2.0):
        super().__init__()

        self.linear_γ = nn.Linear(dim_cond, dim)
        with torch.no_grad():
            nn.init.zeros_(self.linear_γ.weight)
            nn.init.constant_(self.linear_γ.bias, bias_init)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, c, mask = None):
        γ = self.sigmoid(self.linear_γ(c))
        
        if mask is not None:
            γ = γ * mask[..., None]
        
        return x * γ