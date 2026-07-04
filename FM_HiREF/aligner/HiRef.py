import torch
import numpy as np

import operator
import functools
from functools import partial

class HiRefSchedule:
    def __init__(self, max_depth=6, max_terminal_rank=1024, max_rank=500):
        self.max_depth = max_depth
        self.max_terminal_rank = max_terminal_rank
        self.max_rank = max_rank
        self.cached_schedules = {}

    def _get_max_factor(self, n):
        upper = min(self.max_terminal_rank, n)
        for f in range(upper, 0, -1):
            if n % f == 0:
                return f

    def _minimize_factor_product_sum(self, n_to_factor):
        # DP table: stores the minimum cost to reach factor d at depth t
        # Choice table, stores optimal rank for depth t to reach factor d
        dp = np.full((n_to_factor+1, self.max_depth+1), np.inf)
        choice = np.full((n_to_factor+1, self.max_depth+1), -1, dtype=int)

        # Base Case: at depth 1 use initial ranks as cost
        n_initial_ranks = min(n_to_factor, self.max_rank) + 1
        initial_ranks = np.arange(1, n_initial_ranks)

        dp[1:n_initial_ranks, 1] = initial_ranks
        choice[1:n_initial_ranks, 1] = initial_ranks

        # For each depth t, iterate over each rank
        for t in range(2, self.max_depth+1):
            for r in range(1, min(self.max_rank, n_to_factor) + 1):
                # Valid factors d that are multiples of r
                d_vals = np.arange(r, n_to_factor+1, r)
                # Identify the specific state we must have transitioned from t-1
                prev_vals = d_vals // r

                # Calculate the cost of transitioning to 'd_vals' using rank 'r'
                candidate_costs = r + r * dp[prev_vals, t-1] 

                # Update the dp table with the cheaper candidates
                better_mask = candidate_costs < dp[d_vals, t] 
                d_to_update = d_vals[better_mask]
                new_cost = candidate_costs[better_mask]

                dp[d_to_update, t] = new_cost
                choice[d_to_update, t] = r

        # Verifies final state is reached
        if np.isinf(dp[n_to_factor, self.max_depth]):
            return None, []

        # Trace backwards from choice table to build optimal rank schedule
        rank_schedule = []
        d_cur = n_to_factor

        for t_cur in range(self.max_depth, 0, -1):
            r_cur = choice[d_cur, t_cur]
            rank_schedule.append(r_cur)
            d_cur //= r_cur
        
        return dp[n_to_factor, self.max_depth], rank_schedule


    def get_schedule(self, n):
        if n in self.cached_schedules:
            return self.cached_schedules[n]
        
        terminal_rank = self._get_max_factor(n)
        n_to_factor = n // terminal_rank

        _, rank_schedule = self._minimize_factor_product_sum(n_to_factor)
        if not rank_schedule:
            raise ValueError(f"Failed to find a valid factorization schedule for batch size {n}.")

        rank_schedule = sorted(int(x) for x in rank_schedule if x != 1)
        rank_schedule.append(terminal_rank)

        # Validate
        product = functools.reduce(operator.mul, rank_schedule)
        assert product == n, f"Invalid schedule: product={product} != n={n}, schedule={rank_schedule}"

        self.cached_schedules[n] = rank_schedule
        return rank_schedule

class HiRefOT:
    def __init__(self,
                 # OT Parameters
                 base_rank = 1,
                 gamma = 40,
                 rescale_cost = False,
                 iters_per_level = 100,
                 max_sinkhorn_iter=15,
                 # Schedule Paramaters
                 max_depth=6, 
                 max_terminal_rank=1024, 
                 max_rank=500):
        
        self.scheduler = HiRefSchedule(max_depth, max_terminal_rank, max_rank)

        self.base_rank = base_rank
        self.gamma = gamma
        self.rescale = rescale_cost
        self.iters_per_level = iters_per_level
        self.max_sinkhorn_iter = max_sinkhorn_iter

        self.compiled_lrot = torch.compile(self._lrot_lr, dynamic=True)

        self.grad_Q = torch.func.grad(self._loss_lr_two, argnums=0)
        self.grad_R = torch.func.grad(self._loss_lr_two, argnums=1)

    @staticmethod
    def _loss_lr_two(Q, R, A, B, g):
        SA = Q.T @ A
        RB = R.T @ B
        return torch.sum(torch.sum(RB * SA, dim=1) / torch.clamp(g, min=1e-18))

    def _lr_sqeuclidean_factors(self, X, Y):
        N, _ = X.shape
        A = torch.cat([torch.sum(X**2, dim=-1, keepdim=True), torch.ones((N, 1), dtype=X.dtype, device=X.device), (-2 * X)], dim=-1)
        B = torch.cat([torch.ones((N, 1), dtype=Y.dtype, device=Y.device), torch.sum(Y**2, dim=-1, keepdim=True), (Y)], dim=-1)


        if self.rescale:
            sA = torch.sqrt(torch.clamp(torch.max(torch.abs(A)), min=1.0))
            sB = torch.sqrt(torch.clamp(torch.max(torch.abs(B)), min=1.0))
            A = A / sA
            B = B / sB

        return A, B

    @staticmethod
    def _cost_matrix(f, g, G, eps):
        return -(G - f[:, None] - g[None, :]) / eps

    def _log_sinkhorn_projection(self, G, a, b, eps, f=None, g=None, recenter_every=30):
        n, m = G.shape
        
        if f is None: 
            f = torch.zeros((n,), dtype=G.dtype, device=G.device)
        if g is None:
            g = torch.zeros((m,), dtype=G.dtype, device=G.device)

        log_a = torch.log(a)
        log_b = torch.log(b)

        for i in range(self.max_sinkhorn_iter):
            M = self._cost_matrix(f, g, G, eps)
            f = f + eps * (log_a - torch.logsumexp(M, dim=1))
            M = self._cost_matrix(f, g, G, eps)
            g = g + eps * (log_b - torch.logsumexp(M, dim=0))

            if i % recenter_every == 0:
                alpha = torch.mean(f)
                f = f - alpha
                g = g + alpha

        P = torch.exp(self._cost_matrix(f, g, G, eps))
        return P, f, g

    def _initialise_couplings(self, a, b, g, key= None):
        N1, N2, r = a.shape[0], b.shape[0], g.shape[0]
        Cq = torch.rand((N1, r), dtype=a.dtype, device=a.device)
        Cr = torch.rand((N2, r), dtype=b.dtype, device=b.device)
        eps = 1.0 / self.gamma

        # Log Sinkhorn Projection
        Q, _, _ = self._log_sinkhorn_projection(Cq, a, g, eps)
        R, _, _ = self._log_sinkhorn_projection(Cr, b, g, eps)
        return Q, R

    def _md_sinkhorn_step(self, Q, R, A, B, a, b, g, fQ, gQd, fR, gRd):
        gq = self.grad_Q(Q, R, A, B, g)
        gr = self.grad_R(Q, R, A, B, g)

        norm = torch.maximum(torch.max(torch.abs(gq)), torch.max(torch.abs(gr)))
        gamma_k = self.gamma / torch.clamp(norm, min=1e-18)
        eps = 1.0 / gamma_k

        GQ = gq - (1.0 / gamma_k) * torch.log(torch.clamp(Q, min=1e-32))
        GR = gr - (1.0 / gamma_k) * torch.log(torch.clamp(R, min=1e-32))

        Qn, fQn, gQn = self._log_sinkhorn_projection(GQ, a, g, eps, f=fQ, g=gQd)
        Rn, fRn, gRn = self._log_sinkhorn_projection(GR, b, g, eps, f=fR, g=gRd)
        
        return (Qn, Rn, fQn, gQn, fRn, gRn), None

    def _lrot_lr(self, A, B, r, key=None):
        n, m = A.shape[0], B.shape[0]
        a = torch.ones((n,), device=A.device, dtype=A.dtype) / n
        b = torch.ones((m,), device=B.device, dtype=B.dtype) / m
        g = torch.ones((r,), device=B.device, dtype=B.dtype) / r

        Q, R = self._initialise_couplings(a, b, g, key= None)

        fQ = torch.zeros((n,), device=A.device, dtype=A.dtype)
        gQ = torch.zeros((r,), device=A.device, dtype=A.dtype)
        fR = torch.zeros((m,), device=B.device, dtype=B.dtype)
        gR = torch.zeros((r,), device=B.device, dtype=B.dtype)

        for i in range(self.iters_per_level):
            (Q, R, fQ, gQ, fR, gR), _ = self._md_sinkhorn_step(Q, R, A, B, a, b, g, fQ, gQ, fR, gR)

        return Q, R
    
    @staticmethod
    def _split_by_cap(scores, capacity):

        N, RANK = scores.shape
        taken = torch.zeros((N,), dtype=torch.bool, device=scores.device)
        neg_inf = torch.finfo(scores.dtype).min
        neg_inf = torch.tensor(neg_inf, device=scores.device, dtype=scores.dtype)

        out = []
        for r in range(RANK):
            s = torch.where(taken, neg_inf, scores[:, r])
            _, idxs = torch.topk(s, k=capacity, dim=0)
            out.append(idxs)
            taken = taken.index_fill(0, idxs, True)

        return torch.stack(out, dim=0).to(torch.int32)


    def _per_block(self, A, B, rank, cap, idxX, idxY):
        Ai = A[idxX]
        Bi = B[idxY]
        Q, R = self.compiled_lrot(Ai, Bi, rank)

        new_idxX = self._split_by_cap(Q, cap)
        new_idxY = self._split_by_cap(R, cap)

        return new_idxX, new_idxY

    @torch.no_grad()
    def align(self, X, Y):
        X = X.float()
        Y = Y.float()
        
        X_flat = torch.flatten(X, start_dim=1)
        Y_flat = torch.flatten(Y, start_dim=1)

        BATCH_SIZE = X_flat.shape[0]

        # Obtain rank schedule for aligning
        rank_schedule = self.scheduler.get_schedule(BATCH_SIZE)

        # Alignment
        A, B = self._lr_sqeuclidean_factors(X_flat, Y_flat)
        
        current_clusters = [(torch.arange(BATCH_SIZE, device=A.device),
                             torch.arange(BATCH_SIZE, device=B.device))]

        for i, rank in enumerate(rank_schedule):
            work_blocks, leaf_blocks = [], []
            for idxX, idxY in current_clusters:
                if min(int(idxX.shape[0]), int(idxY.shape[0])) <= self.base_rank:
                    leaf_blocks.append((idxX, idxY))
                else:
                    work_blocks.append((idxX, idxY))

            new_clusters = list(leaf_blocks)

            # Vectorising the work blocks
            if work_blocks:
                stacked_idxX = torch.stack([x for x, _ in work_blocks])
                stacked_idxY = torch.stack([y for _, y in work_blocks])                
                
                # Check capacity
                block_size = stacked_idxX.shape[1]
                cap = int(block_size // rank)
                if cap <= 0:
                    new_clusters.extend(work_blocks)
                    current_clusters = new_clusters
                    continue

                # Run Low Rank Optimal Transport across all blocks simultaneously
                per_block = partial(self._per_block, A, B, rank, cap)
                Xi, Yi = torch.vmap(per_block, randomness="different")(stacked_idxX, stacked_idxY)
                
                # Unpack the batched operation
                num_blocks = stacked_idxX.shape[0]
                batch_idx = torch.arange(num_blocks, device=stacked_idxX.device).view(-1, 1, 1)

                Xi_global = stacked_idxX[batch_idx, Xi]
                Yi_global = stacked_idxY[batch_idx, Yi]

                Xi_flat = Xi_global.view(-1, cap)
                Yi_flat = Yi_global.view(-1, cap)
                new_clusters.extend(zip(Xi_flat, Yi_flat))

            current_clusters = new_clusters
        
        # Align Noise with X
        flat_X_idx = torch.cat([c[0] for c in current_clusters])
        flat_Y_idx = torch.cat([c[1] for c in current_clusters])
        
        sort_order = torch.argsort(flat_X_idx)
        aligned_Y_idx = flat_Y_idx[sort_order]

        return Y[aligned_Y_idx], current_clusters