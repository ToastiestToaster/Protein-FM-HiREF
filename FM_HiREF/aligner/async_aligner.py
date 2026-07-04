import threading
import multiprocessing as mp
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment

from FM_HiREF.aligner.base_aligner import BaseAligner

def _alignment_worker(args):
    torch.set_num_threads(1)
    cluster_inds, global_inds, X, alignment_threshold, aligner, epoch_seed = args
    
    # Initialise aligner
    aligner = aligner()

    # Set seed
    gen = None
    if epoch_seed is not None:
        cluster_seed = epoch_seed + global_inds[0] 
        gen = torch.Generator(device='cpu').manual_seed(cluster_seed)

    # Generate noise
    N = len(cluster_inds)
    Y = torch.randn(*X.shape, generator=gen, device=X.device, dtype=X.dtype)
    Y = Y - torch.mean(Y, dim=-2, keepdim=True)

    # Alignment logic
    if N == 1:
        aligned_Y = Y
    elif N <= alignment_threshold:
        X_flat = torch.flatten(X, start_dim=1)
        Y_flat = torch.flatten(Y, start_dim=1)
        C = torch.cdist(X_flat, Y_flat, p=2) ** 2
        _, col_ind = linear_sum_assignment(C.numpy())
        aligned_Y = Y[col_ind]
    else:
        aligned_Y, _ = aligner.align(X, Y)
    
    return global_inds, aligned_Y

class AsyncAligner(BaseAligner):
    def __init__(self, alignment_threshold, max_workers, aligner):
        super().__init__(alignment_threshold, aligner)
        self.alignment_threshold = alignment_threshold
        self.max_workers = max_workers
        self.aligner = aligner

        self.thread = None
        self.next_noise_dict = {}    

    def load_dataset(self, dataset):
        # Cache dataset coordinates in RAM and prepare static features.
        self.base_task_data = []

        for cluster_id, cluster_inds in dataset.clusters.items():
            X = torch.stack([dataset[idx]['atom_CA'] for idx in cluster_inds]).cpu()
            global_inds = [dataset.indices[c_idx] for c_idx in cluster_inds]
            
            self.base_task_data.append((cluster_inds, 
                                        global_inds, 
                                        X, 
                                        self.alignment_threshold,
                                        self.aligner,))
    
    def begin_alignment(self, seed):
        self.next_noise_dict = {}

        # Contains the args for each job
        tasks = [(c_inds, g_inds, X, thresh, aligner, seed)
                 for (c_inds, g_inds, X, thresh, aligner) in self.base_task_data]
        
        self.thread = threading.Thread(target=self._run_process_pool, args=(tasks,))
        self.thread.start()
        
    def _run_process_pool(self, tasks):
        ctx = mp.get_context("spawn")
        
        with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_alignment_worker, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    global_inds, aligned_Y = future.result()
                    for i, global_idx in enumerate(global_inds):
                        self.next_noise_dict[global_idx] = aligned_Y[i]

                except Exception as e:
                    # Save the exact error so the main thread can see it (Debugging Purposes)
                    self.fatal_error = e
                    raise

    def fetch_aligned_noise(self):
        if self.thread is not None:
            self.thread.join()
        
        # Check if an error has occured in the process pool
        if hasattr(self, 'fatal_error'):
            raise RuntimeError("Background alignment failed!") from self.fatal_error   
             
        return self.next_noise_dict