import torch
import torch.nn as nn
import math

from .networks import (CondFeatureNet,
                       SingleFeatureNet,
                       PairFeatureNet,
                       AdaptiveStructureNet)

def embed_t(t, emb_dim, max_positions=2000):
    if emb_dim % 2 != 0:
        raise ValueError(f"Embedding t requires an even 'dim' to split between sine and cosine, but received dim={emb_dim}.")
    
    t_scaled = t.float() * max_positions
    half_dim = emb_dim // 2
    emb_scale = math.log(max_positions) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_scale)

    emb = t_scaled[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    return emb

class FlowMatchingNetwork(nn.Module):
    def __init__(self,
                 # Conditioning Feature Parameters
                 t_embed_dim,
                 c_dim,
                 c_factor,
                 
                 # Single Feature Parameters
                 idx_emb_dim,
                 token_dim,

                 # Pair Feature Parameters
                 relpos_dim, 
                 xt_pair_dist_dim, 
                 xt_pair_dist_min, 
                 xt_pair_dist_max, 
                 pair_dim,
                 
                 # Structure Module Parameters
                 structure_layers,
                 ipa_hidden_dim,
                 ipa_heads,
                 qk_points,
                 v_points,):
        super().__init__()
        self.t_embed_dim = t_embed_dim

        # Conditioning Network
        self.c_embedder = CondFeatureNet(t_embed_dim,
                                         c_dim,
                                         c_factor)

        # Single Feature Network <- More here when moving into multi-modal flow matching
        self.s_embedder = SingleFeatureNet(idx_emb_dim,
                                           token_dim)

        # Pair Feature Network
        self.p_embedder = PairFeatureNet(relpos_dim, 
                                         xt_pair_dist_dim, 
                                         xt_pair_dist_min, 
                                         xt_pair_dist_max, 
                                         pair_dim, 
                                         t_embed_dim, 
                                         c_dim)
        
        # Structure Network
        self.structure_net = AdaptiveStructureNet(structure_layers,
                                                  token_dim,
                                                  c_dim,
                                                  pair_dim,
                                                  ipa_hidden_dim,
                                                  ipa_heads,
                                                  qk_points,
                                                  v_points)

    def forward(self, x_t, f_t, t, res_id, mask=None):
        # Embed t
        t_emb = embed_t(t, self.t_embed_dim)

        # Generate Features
        c = self.c_embedder(t_emb)
        s = self.s_embedder(res_id)
        p = self.p_embedder(x_t, t_emb, res_id, mask)
        
        # Predict frames at t=1
        states, pred_f_1 = self.structure_net(s, c, p, f_t, mask)

        # Generate the vector field
        v_t_pred = (pred_f_1.t - x_t) / (1.0 - t[..., None, None] + 1e-5)
        if mask is not None:
            v_t_pred = v_t_pred * mask[..., None]
        
        return {"pred_frames": pred_f_1, 
                "pred_x_1": pred_f_1.t, 
                "states": states,
                "v_t_pred": v_t_pred,}