import lightning as L
from torch.utils.data import DataLoader, Subset, Sampler, RandomSampler
from collections import defaultdict
import random

from FM_HiREF.data.dataset import PDBDataset

def max_prime_factor(n):
    if n <= 1: return 1
    max_prime = -1
    while n % 2 == 0:
        max_prime = 2
        n >>= 1
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            max_prime = i
            n //= i
    if n > 2:
        max_prime = n
    return max_prime

def is_smooth(n, max_factor=10):
    if n <= 0:
        return True
    return max_prime_factor(n) <= max_factor

def find_best_smooth_triplet(total_N, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, max_factor=5, search_window=30):
    if total_N <= 2:
        return total_N, 0, 0

    ideal_t = int(total_N * train_ratio)
    ideal_v = int(total_N * val_ratio)
    
    # Initialise triplet scores
    best_sum = -1
    best_ratio_error = float('inf')
    best_triplet = (0, 0, 0)
    
    # Establish search space
    t_min, t_max = max(0, ideal_t - search_window), min(total_N, ideal_t + search_window)
    v_min, v_max = max(0, ideal_v - search_window), min(total_N, ideal_v + search_window)
    
    # Selects all valid smooth composite numbers in the search space
    valid_t = [t for t in range(t_min, t_max + 1) if is_smooth(t, max_factor)]
    valid_v = [v for v in range(v_min, v_max + 1) if is_smooth(v, max_factor)]
    
    # Explore each combination of t and v
    for t in valid_t:
        for v in valid_v:
            remaining = total_N - t - v
            # If less than 0 is left over, move onto next smooth v
            if remaining < 0: 
                continue
            
            # Cycle through remaining smooth values
            for e in range(remaining, -1, -1):
                if is_smooth(e, max_factor):
                    current_sum = t + v + e

                    # Calculate ratios of this specific combination + Distance from desired combination
                    r_t, r_v, r_e = t / current_sum, v / current_sum, e / current_sum
                    error = (r_t - train_ratio)**2 + (r_v - val_ratio)**2 + (r_e - test_ratio)**2

                    # First priority: maximise the datapoints in the set
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_ratio_error = error
                        best_triplet = (t, v, e)
                    # Second priority: minimise error from original train/val/test split
                    elif current_sum == best_sum and error < best_ratio_error:
                        best_ratio_error = error
                        best_triplet = (t, v, e)
                    break 
                    
    return best_triplet

class ClusteredSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.clusters = defaultdict(list)
        for local_idx, global_idx in enumerate(indices):
            length = self.dataset.lengths[global_idx]
            self.clusters[length].append(local_idx)

class LengthBatchSampler(Sampler):
    def __init__(self, data_source : ClusteredSubset, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.clusters = data_source.clusters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __iter__(self):
        batches = []
        for indices in self.clusters.values():
            cluster_indices = list(indices)

            if self.shuffle:
                random.shuffle(cluster_indices)
            
            for i in range(0, len(cluster_indices), self.batch_size):
                batch = cluster_indices[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)
    
    def __len__(self):
        total_batches = 0
        for indices in self.clusters.values():
            if self.drop_last:
                # Math for dropping (rounds down)
                total_batches += len(indices) // self.batch_size
            else:
                # Math for keeping (rounds up)
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches

class PDBDataModule(L.LightningDataModule):
    def __init__(self, pdb_dir, scale, train_val_test_split, batch_size, max_factor, num_workers=1, persistent_workers=True, seed=None):
        super().__init__()
        self.pdb_dir = pdb_dir
        self.scale = scale
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.max_factor = max_factor
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.seed = seed
    
    def prepare_data(self):
        # Caches the data
        PDBDataset.cache_pdbs(self.pdb_dir, self.scale)
    
    def _distribute_indices(self, dataset):
        rng = random.Random(self.seed)
        train_idx, val_idx, test_idx = [], [], []

        for _, indices in dataset.length_clusters.items():
            cluster_indices = list(indices)
            rng.shuffle(cluster_indices)
            total_N = len(cluster_indices)

            if total_N <= 2: 
                train_idx.extend(cluster_indices)
                continue
            
            train_N, val_N, test_N = find_best_smooth_triplet(total_N, *self.train_val_test_split, self.max_factor, search_window=30)

            # Assign the indices for the cluster.
            idx_ptr = 0
            train_idx.extend(cluster_indices[idx_ptr : idx_ptr + train_N])
            idx_ptr += train_N
            val_idx.extend(cluster_indices[idx_ptr : idx_ptr + val_N])
            idx_ptr += val_N
            test_idx.extend(cluster_indices[idx_ptr : idx_ptr + test_N])

        return train_idx, val_idx, test_idx

    def setup(self, stage=None):
        self.dataset = PDBDataset(self.pdb_dir, self.scale)
        
        # Dynamically split, maintaining non-prime cluster sizes in training and validation sets.
        train_idx, val_idx, test_idx = self._distribute_indices(self.dataset)

        # Datasets made using modified Subset class
        self.train_ds = ClusteredSubset(self.dataset, train_idx)
        self.val_ds = ClusteredSubset(self.dataset, val_idx)
        self.test_ds = ClusteredSubset(self.dataset, test_idx)

        # Initialise samplers
        self.train_sampler = LengthBatchSampler(self.train_ds, self.batch_size, drop_last=True)
        self.val_sampler = LengthBatchSampler(self.val_ds, self.batch_size, shuffle=False, drop_last=False)
        self.test_sampler = LengthBatchSampler(self.test_ds, self.batch_size, shuffle=False, drop_last=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_sampler=self.train_sampler, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_sampler=self.val_sampler, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_sampler=self.test_sampler, num_workers=self.num_workers, persistent_workers=self.persistent_workers)