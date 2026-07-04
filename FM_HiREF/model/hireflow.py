import torch

import lightning as L
import torch.nn as nn

from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from FM_HiREF.aligner.base_aligner import BaseAligner
from FM_HiREF.aligner.HiRef import HiRefOT
from FM_HiREF.protein.frames import Frames

import time

class HiReFlow(L.LightningModule):
    def __init__(self, 
                 # Networks
                 net: nn.Module, 
                 
                 # Trainer Parameters
                 warmup_epochs,
                 max_epochs,
                 lr: float = 1e-4,
                 use_scheduler: bool = False,
                 compile: bool = False,
                 
                 # For the Async Aligner
                 aligner: BaseAligner = None,):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['aligner', 'net'])

        # Model Framework
        self.net = net

        # Noise
        self.val_noise = None
        self.train_noise = None
        self.test_noise = None

        # Aligners - Passed through as partials
        self.train_aligner = aligner() if aligner is not None else None
        self.val_aligner   = aligner() if aligner is not None else None
        self.test_aligner  = aligner() if aligner is not None else None

        # Losses
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_best = MinMetric()

    def _batched_kabsch(self, P, Q, mask):
        mask_w = mask[..., None] # [B, N, 1]

        # Compute Centers of Mass minimizing padding token corruption
        p_com = torch.sum(P * mask_w, dim=1, keepdim=True) / (torch.sum(mask_w, dim=1, keepdim=True) + 1e-5)
        q_com = torch.sum(Q * mask_w, dim=1, keepdim=True) / (torch.sum(mask_w, dim=1, keepdim=True) + 1e-5)

        P_centered = (P - p_com) * mask_w
        Q_centered = (Q - q_com) * mask_w

        # Compute Cross-Covariance Matrix
        H = torch.bmm(P_centered.transpose(-1, -2), Q_centered) # [B, 3, 3]

        # Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(H)

        # Calculate Rotation Matrix and correct for reflections
        R = torch.bmm(Vh.transpose(-1, -2), U.transpose(-1, -2))
        is_reflection = torch.linalg.det(R) < 0
            
        if torch.any(is_reflection):
            E = torch.eye(3, device=P.device).repeat(P.shape[0], 1, 1)
            E[is_reflection, 2, 2] = -1.0
            R = torch.bmm(Vh.transpose(-1, -2), torch.bmm(E, U.transpose(-1, -2)))

        # Rotate centered source points and translate to target frame
        P_aligned = torch.bmm(P_centered, R.transpose(-1, -2)) + q_com
        return P_aligned * mask_w


    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()
        self.val_best.reset()

        if self.train_aligner is not None:
            self.train_aligner.load_dataset(self.trainer.datamodule.train_ds)
            self.global_run_seed = torch.initial_seed()
            self.train_aligner.begin_alignment(self.global_run_seed)

    def on_train_epoch_start(self):
        if self.train_aligner is not None:
            self.train_noise = self.train_aligner.fetch_aligned_noise()

            # Begin generating noise for the next epoch
            #   - Required for asynchronous aligner classes
            next_epoch_seed = self.global_run_seed + self.current_epoch + 1
            self.train_aligner.begin_alignment(next_epoch_seed)
        else:
            self.train_noise = None

    def on_validation_start(self):
        # Static validation noise is only generated once
        if self.val_aligner is not None and self.val_noise is None:
            self.val_aligner.load_dataset(self.trainer.datamodule.val_ds)
            self.val_aligner.begin_alignment(1)
            self.val_noise = self.val_aligner.fetch_aligned_noise()

    def on_test_start(self):
        # Static test noise is only generated once at test time using a stable test seed
        if self.test_aligner is not None and self.test_noise is None:
            self.test_aligner.load_dataset(self.trainer.datamodule.test_ds)
            self.test_aligner.begin_alignment(1)
            self.test_noise = self.test_aligner.fetch_aligned_noise()

    def training_step(self, batch):
        loss = self.model_step(batch, self.train_noise)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = self.model_step(batch, self.val_noise)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.val_best(val_loss)
        self.log("val/loss_best", self.val_best.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch):
        loss = self.model_step(batch, self.test_noise)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def model_step(self, batch, noise_dict):        
        res_id = batch['res_id']
        mask = batch['atom_CA_mask']
        x_1 = batch['atom_CA']
        if noise_dict is not None:
            # Extract pre-aligned optimal transport noise tracks from cache
            x_0 = torch.stack([noise_dict[l.item()] for l in batch['label']]).to(self.device)
        else:
            # Dynamically generate standard zero-centered Gaussian noise if no aligner
            x_0 = torch.randn_like(x_1)
            x_0 = x_0 - torch.mean(x_0, dim=-2, keepdim=True)

        B_SIZE, N_RES = mask.shape

        # Sample timestep
        t = (torch.rand(1, device=self.device) + torch.arange(B_SIZE, device=self.device)) / B_SIZE

        # Interpolate
        _t = t[..., None, None]
        x_t = (1.0 - _t) * x_0 + _t * x_1
        x_t = x_t * mask[..., None]

        # Build frames object for x_t
        f_t = Frames.from_frenet_serret(x_t)
        
        model_out = self.forward(x_t, f_t, t, res_id, mask)

        # Coordinate loss
        x_1_pred = model_out['pred_x_1'] * mask[..., None]
        x_1_target = x_1 * mask[..., None]
        # x_1_pred_aligned = self._batched_kabsch(x_1_pred, x_1_target, mask)

        coord_mse = torch.sum((x_1_pred - x_1_target) ** 2, dim=-1)
        loss_coords = torch.sum(coord_mse, dim=-1) / (torch.sum(mask, dim=-1) + 1e-5)

        loss = torch.mean(loss_coords)
        return loss

    def setup(self, stage):
        if self.hparams.compile and (stage == "fit" or stage is None):
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr)

        if not self.hparams.use_scheduler:
            return {"optimizer": optimizer}
        
        warmup_iters = self.hparams.warmup_epochs
        decay_iters = self.hparams.max_epochs - warmup_iters

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                   start_factor=0.001, 
                                                   end_factor=1.0, 
                                                   total_iters=warmup_iters)
        
        
        decay = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                  start_factor=1.0, 
                                                  end_factor=0.002, 
                                                  total_iters=decay_iters)
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                          schedulers=[warmup, decay], 
                                                          milestones=[warmup_iters])
        
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": "epoch",
                                 "frequency": 1,},}
    
    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]

        for key in list(state_dict.keys()):
            if key.startswith("net._orig_mod."):
                new_key = key.replace("net._orig_mod.", "net.", 1)
                state_dict[new_key] = state_dict.pop(key) 

        return super().on_save_checkpoint(checkpoint)

    def forward(self, x_t, f_t, t, res_id, mask):
        return self.net(x_t, f_t, t, res_id, mask)
    
    @torch.no_grad()
    def sample_protein(self, batch_size, num_res, scale: int, num_steps: int = 50, save_trajectory: bool = False, seed: int = None):    
        self.eval()

        # Seed
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        # Initialise x_0
        x_t = torch.randn(batch_size, num_res, 3, generator=gen, device=self.device)
        x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)
        mask = torch.ones(batch_size, num_res, device=self.device)

        # model inputs
        dt = (1.0 / num_steps)
        timesteps = torch.linspace(0.0, 1.0 - dt, num_steps, device=self.device)
        res_id = torch.arange(1, num_res + 1, device=self.device).expand(batch_size, -1)

        # Storing the trajectory
        trajectory = []
        if save_trajectory:
            trajectory.append(x_t.clone().cpu())

        for t_val in tqdm(timesteps, desc=f"Folding {batch_size} proteins ({num_res} res)"):            
            # model inputs
            t = torch.full((batch_size,), t_val, device=self.device)
            f_t = Frames.from_frenet_serret(x_t)

            # Model Pass
            model_out = self.forward(x_t, f_t, t, res_id, mask)
            v_t = model_out['v_t_pred']

            x_t = x_t + (v_t * dt)
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)
            
            # Add to trajectory list
            if save_trajectory:
                trajectory.append(x_t.clone().cpu())
        
        x_1 = x_t * scale

        if save_trajectory:
            # Stack the trajectory tensor list and scale the entire block at once
            trajectory = torch.stack(trajectory, dim=1) * scale
            return x_1, trajectory
        
        return x_1