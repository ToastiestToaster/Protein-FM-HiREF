import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.adaln import AdaptiveLayerNorm, AdaptiveOutputScale
from .modules.transition import Transition
from .modules.ipa import InvariantPointAttention
from .modules.bb_update import BackboneUpdate

# Embedding Functions
def embed_res_indices(indices, emb_dim, max_len=2056):
    if emb_dim % 2 != 0:
        raise ValueError(f"Embedding t requires an even 'dim' to split between sine and cosine, but received dim={emb_dim}.")

    half_dim = emb_dim // 2
    K = torch.arange(half_dim, dtype=torch.float32, device=indices.device)
    
    inv_freq = math.pi / (max_len ** (2 * K / emb_dim))
    phase = indices[..., None].float() *  inv_freq
    pos_embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

    return pos_embedding

def bin_and_one_hot(tensor, bins):
    bindices = torch.bucketize(tensor, bins)
    return F.one_hot(bindices, len(bins) + 1).float()

def relpos_from_indices(indices, seq_sep_dim):
    # seq_sep_dim should be odd and more than or equal to 5

    relpos = indices[..., None] - indices[..., None, :]
    bound = seq_sep_dim / 2.0 - 1
    bindices = torch.linspace(-bound, bound, seq_sep_dim - 1, device=indices.device)

    return bin_and_one_hot(relpos, bindices)

def bin_pairwise_distances(ca_coors, l_bound, u_bound, dim):
    pair_dists = torch.norm(ca_coors[..., None, :] - ca_coors[..., None, :, :], dim=-1)
    bindices = torch.linspace(l_bound, u_bound, dim - 1, device=ca_coors.device)
    return bin_and_one_hot(pair_dists, bindices)

# Feature Networks
class CondFeatureNet(nn.Module):
    def __init__(self, t_embed_dim, c_dim, factor):
        super().__init__()

        self.linear_c = torch.nn.Linear(t_embed_dim, c_dim, bias=False)
        self.transition_1 = Transition(c_dim, factor)
        self.transition_2 = Transition(c_dim, factor)
    
    def forward(self, t_emb):
        c = self.linear_c(t_emb)
        c = c[..., None, :]
        c = self.transition_2(self.transition_1(c))
        return c

class PairFeatureNet(nn.Module):
    def __init__(self, 
                 # Pair Features
                 relpos_dim, 
                 pair_dist_dim,
                 pair_dist_min, 
                 pair_dist_max,
                 
                 # Network Layers
                 pair_dim,
                 t_embed_dim,
                 c_pair_dim):
        super().__init__()

        # Process pair features first
        self.p_ln = torch.nn.LayerNorm(pair_dim)
        self.p_linear = torch.nn.Linear((relpos_dim + pair_dist_dim), pair_dim, bias=False)
        self.relpos_dim = relpos_dim
        self.pair_dist_dim = pair_dist_dim
        self.pair_dist_min = pair_dist_min
        self.pair_dist_max = pair_dist_max

        # Time Embedding (Conditioning)
        self.c_pair_linear = torch.nn.Linear(t_embed_dim, c_pair_dim, bias=False)
        self.c_pair_ln = torch.nn.LayerNorm(c_pair_dim)

        # Adaln
        self.adaln = AdaptiveLayerNorm(pair_dim, c_pair_dim)

    def forward(self, x_t, t_emb, res_id, mask):

        # Build mask
        pair_mask = mask[..., None] * mask[..., None, :]

        # Pair feats
        relpos = relpos_from_indices(res_id, self.relpos_dim)
        pair_dist = bin_pairwise_distances(x_t, self.pair_dist_min, self.pair_dist_max, self.pair_dist_dim)
        p_feat = torch.cat([relpos, pair_dist], dim=-1)
        p_feat = self.p_ln(self.p_linear(p_feat))

        # Conditioning (Time) Feat
        c = self.c_pair_ln(self.c_pair_linear(t_emb))
        c = c[..., None, None, :]

        return self.adaln(p_feat, c, pair_mask)


class SingleFeatureNet(nn.Module):
    def __init__(self, idx_emb_dim, s_dim):
        super().__init__()
        self.idx_emb_dim = idx_emb_dim
        self.s_linear = torch.nn.Linear(idx_emb_dim, s_dim, bias=False)
    
    def forward(self, res_id):
        
        # Single feature
        res_id = res_id - res_id[:, 0][..., None] + 1
        res_emb = embed_res_indices(res_id, self.idx_emb_dim)
        
        return self.s_linear(res_emb)

# Structure Networks
class AdaptiveStructureLayer(nn.Module):
    def __init__(self, 
                 # Feature Dims
                 s_dim, 
                 c_dim, 
                 p_dim,
                 
                 # IPA Paramters
                 ipa_hidden_dim,
                 ipa_heads,
                 qk_points,
                 v_points):
        super().__init__()

        # Adaptive Invariant Point Attention
        self.adaln_1 = AdaptiveLayerNorm(s_dim, c_dim)
        self.ipa = InvariantPointAttention(s_dim, p_dim, ipa_hidden_dim, ipa_heads, qk_points, v_points)
        self.ada_scale_1 = AdaptiveOutputScale(s_dim, c_dim)
        
        # Adaptive Transition
        self.adaln_2 = AdaptiveLayerNorm(s_dim, c_dim)
        self.transition = Transition(s_dim, 4)
        self.ada_scale_2 = AdaptiveOutputScale(s_dim, c_dim)

        # Backbone update
        self.bb_update = BackboneUpdate(s_dim)
    
    def forward(self, s, c, p, f_t, mask=None):
        
        # Adaptive IPA
        ds = self.adaln_1(s, c, mask)
        ds = self.ipa(ds, p, f_t, mask)
        s = s + self.ada_scale_1(ds, c, mask)

        # Adaptive Transition
        ds = self.adaln_2(s, c, mask)
        ds = self.transition(ds)
        s = s + self.ada_scale_2(ds, c, mask)

        # Backbone update
        update_frame = self.bb_update(s)
        updated_frame = f_t.compose(update_frame)

        return updated_frame, s

class AdaptiveStructureNet(nn.Module):
    def __init__(self,
                 # Network Parameters
                 n_layers,

                 # Feature Dims
                 s_dim, 
                 c_dim, 
                 p_dim,
                 
                 # IPA Paramters
                 ipa_hidden_dim,
                 ipa_heads,
                 qk_points,
                 v_points,):
        super().__init__()

        self.structure_net = nn.ModuleList([AdaptiveStructureLayer(s_dim, 
                                                                   c_dim, 
                                                                   p_dim,
                                                                   
                                                                   # IPA Paramters
                                                                   ipa_hidden_dim,
                                                                   ipa_heads,
                                                                   qk_points,
                                                                   v_points) for _ in range(n_layers)])
    
    def forward(self, s, c, p, f_t, mask=None):

        states = [s]
        curr_f_t = f_t
        for layer in self.structure_net:
            curr_f_t, s = layer(s, c, p, curr_f_t, mask)
            states.append(s)

        return states, curr_f_t