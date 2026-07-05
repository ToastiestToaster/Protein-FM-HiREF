import torch
import torch.nn as nn
import torch.nn.functional as F

class InvariantPointAttention(nn.Module):
    def __init__(self, single_dim, pair_dim, hidden_dim, heads, qk_points, v_points):
        super().__init__()
        self.inf = 1e5
        
        # Normalisation parameters
        self.norm = hidden_dim**(-0.5)
        self.wc = (2 / (9 * qk_points)) ** (0.5)
        self.wl = 3**(-0.5)
        self.γ = nn.Parameter(torch.zeros(heads)) # Trainable parameter per head
        with torch.no_grad():
            self.γ.fill_(0.541324854612918)

        # Initialise attention parameters
        self.heads = heads
        self.qk_points = qk_points
        self.v_points = v_points
        self.hidden_dim = hidden_dim

        # Query, Key, Value and Bias linear layers
        hc = self.heads * self.hidden_dim
        self.linear_q = nn.Linear(single_dim, hc)
        self.linear_k = nn.Linear(single_dim, hc)
        self.linear_v = nn.Linear(single_dim, hc)
        self.linear_b = nn.Linear(pair_dim, heads)

        # Query point, Key point and Value point linear layers
        self.linear_qp = nn.Linear(single_dim, (3 * qk_points * heads))
        self.linear_kp = nn.Linear(single_dim, (3 * qk_points * heads))
        self.linear_vp = nn.Linear(single_dim, (3 * v_points * heads))

        # Linear pair down, since it is not processed before hand but probably could?
        self.linear_pair_down = nn.Linear(pair_dim, pair_dim // 4)

        # Initialise Linear Output Layer as zeros (final)
        out_dims = heads * (pair_dim // 4 + hidden_dim + v_points * 4)
        self.linear_o = nn.Linear(out_dims, single_dim)
        with torch.no_grad():
            nn.init.zeros_(self.linear_o.weight)
            nn.init.zeros_(self.linear_o.bias)

    def forward(self, single, pair, frames, seq_mask=None): # INCLUDE MASKS
        # Process Single Representation into qkv
        q = self.linear_q(single)
        q = q.view(q.shape[:-1] + (self.heads, self.hidden_dim)) * self.norm
        
        k = self.linear_k(single)
        k = k.view(k.shape[:-1] + (self.heads, self.hidden_dim))
        
        v = self.linear_v(single)
        v = v.view(v.shape[:-1] + (self.heads, self.hidden_dim))
        
        # Process Single Representation into qkv points
        qp = self.linear_qp(single)
        qp = qp.view(qp.shape[:-1] + (self.heads, self.qk_points, 3))
        qp = frames[..., None, None].apply(qp)

        kp = self.linear_kp(single)
        kp = kp.view(kp.shape[:-1] + (self.heads, self.qk_points, 3))
        kp = frames[..., None, None].apply(kp)

        vp = self.linear_vp(single)
        vp = vp.view(vp.shape[:-1] + (self.heads, self.v_points, 3))
        vp = frames[..., None, None].apply(vp)

        # Pair to bias
        b = self.linear_b(pair)

        # Compute attention weights
        attn = torch.einsum('...ihc,...jhc->...ijh', q, k)
        attn = attn + b

        broadcast = (None,) * len(attn.shape[:-1]) + (slice(None), None)
        γ = nn.functional.softplus(self.γ)[broadcast]

        p_attn = qp.unsqueeze(-4) - kp.unsqueeze(-5)
        p_attn = (p_attn ** 2).sum(dim=(-1))
        p_attn = (γ * self.wc / 2) * p_attn
        p_attn = p_attn.sum(dim=-1)


        attn = self.wl * (attn - p_attn)
        if seq_mask is not None:
            mask = seq_mask[..., None] * seq_mask[..., None, :]
            mask = (mask[..., None] - 1) * self.inf
            attn = attn + mask
        attn = nn.functional.softmax(attn, dim=-2)

        # Compute outputs
        pair_down = self.linear_pair_down(pair)
        o1 = torch.einsum('...ijh,...ijc->...ihc', attn, pair_down)
        o2 = torch.einsum('...ijh,...jhc->...ihc', attn, v)
        o3 = torch.einsum('...ijh,...jhpx->...ihpx', attn, vp)
        o3 = frames[..., None, None].invert_apply(o3)
        o4 = torch.norm(o3, dim=-1)

        # Reshape outputs
        o1 = o1.reshape(o1.shape[:-2] + (-1,))
        o2 = o2.reshape(o2.shape[:-2] + (-1,))
        o3 = o3.reshape(o3.shape[:-3] + (-1, 3))
        o4 = o4.reshape(o4.shape[:-2] + (-1,))

        o = torch.cat([o1, o2, *torch.unbind(o3, dim=-1), o4], dim=-1)
        o = self.linear_o(o)
        return o
