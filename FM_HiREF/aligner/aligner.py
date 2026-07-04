import torch
from scipy.optimize import linear_sum_assignment

from FM_HiREF.aligner.base_aligner import BaseAligner

class Aligner(BaseAligner):
    def __init__(self, alignment_threshold, aligner):
        super().__init__(alignment_threshold, aligner)
        self.alignment_threshold = alignment_threshold
        self.aligner = aligner  
        
        self.base_task_data = []
        self.noise_dict = {}

    def load_dataset(self, dataset):
        self.base_task_data = []
        for cluster_id, cluster_inds in dataset.clusters.items():
            X = torch.stack([dataset[idx]['atom_CA'] for idx in cluster_inds]).cpu()
            global_inds = [dataset.indices[c_idx] for c_idx in cluster_inds]
            self.base_task_data.append((cluster_inds, global_inds, X))

    def begin_alignment(self, seed):
        self.noise_dict = {}
        
        for cluster_inds, global_inds, X in self.base_task_data:
            
            gen = None
            if seed is not None:
                gen = torch.Generator(device='cpu').manual_seed(seed + global_inds[0])

            N = len(cluster_inds)
            Y = torch.randn(*X.shape, generator=gen, device=X.device, dtype=X.dtype)
            Y = Y - torch.mean(Y, dim=-2, keepdim=True)

            if N == 1:
                aligned_Y = Y
            elif N <= self.alignment_threshold:
                X_flat = torch.flatten(X, start_dim=1)
                Y_flat = torch.flatten(Y, start_dim=1)
                C = torch.cdist(X_flat, Y_flat, p=2) ** 2
                _, col_ind = linear_sum_assignment(C.numpy())
                aligned_Y = Y[col_ind]
            else:
                aligned_Y, _ = self.aligner.align(X, Y)
            
            # Populate noise dict
            for i, global_idx in enumerate(global_inds):
                self.noise_dict[global_idx] = aligned_Y[i]

    def fetch_aligned_noise(self):
        return self.noise_dict