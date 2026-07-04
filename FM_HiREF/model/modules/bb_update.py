import torch
import torch.nn as nn

from FM_HiREF.protein.frames import Frames

class BackboneUpdate(nn.Module):
    def __init__(self, single_dim):
        super().__init__()

        # Linear Layer
        self.linear_update = nn.Linear(single_dim, 6, bias=False)

        # Zero initialise
        with torch.no_grad():
            nn.init.zeros_(self.linear_update.weight)
    
    def forward(self, single):
        upd = self.linear_update(single)
        abcd = torch.cat([upd.new_ones(*upd.shape[:-1], 1), upd[..., :3]], dim=-1)
        a, b, c, d = ( abcd /  abcd.norm(dim=-1, keepdim=True) ).unbind(dim=-1)

        # Quaternion to Rotation Matrix [...,] -> [..., 3] x3 -> [..., 3, 3]
        row1 = torch.stack([ a**2 + b**2 - c**2 - d**2 , 2*b*c - 2*a*d , 2*b*d + 2*a*c ], dim=-1) # [..., 3]
        row2 = torch.stack([ 2*b*c + 2*a*d , a**2 - b**2 + c**2 - d**2 , 2*c*d - 2*a*b ], dim=-1)
        row3 = torch.stack([ 2*b*d - 2*a*c , 2*c*d + 2*a*b , a**2 - b**2 - c**2 + d**2 ], dim=-1)

        # Stack again, as these are rows, we want to stack them in the second land dimension
        rots = torch.stack([row1, row2, row3], dim=-2)
        trans = upd[..., 3:]

        return Frames(rots, trans)