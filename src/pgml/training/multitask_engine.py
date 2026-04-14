from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class MultiTaskStateEstimationEngine(L.LightningModule):
    """
    Training engine for the first hierarchical explicit-device architecture.

    Loss split:
    - encoder/decoder-style reconstruction losses
    - estimator losses

    In this first implementation, these are numerically identical because the
    graph estimator is part of the same pathway and targets are direct token
    reconstructions. The logging split is still introduced now so later training
    phases can separate them cleanly.

    #TODO: Split local autoencoding losses from graph-estimation losses more
    #      explicitly once a pretraining stage or separate local decoders exist.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha_recon: float = 1.0,
        beta_state: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha_recon = alpha_recon
        self.beta_state = beta_state

        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(batch)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device if pred.numel() > 0 else target.device)

        mask_f = mask.unsqueeze(-1).to(pred.dtype)
        diff_sq = ((pred - target) ** 2) * mask_f
        denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
        return diff_sq.sum() / denom

    def _shared_step(self, batch, phase: str):
        outputs = self(batch)

        loss_node = self._masked_mse(
            outputs["pred_node_value"],
            batch["target_node"].voltage_value,
            batch["target_node"].voltage_mask,
        )

        loss_edge = self._masked_mse(
            outputs["pred_edge_value"],
            batch["target_edge"].current_value,
            batch["target_edge"].current_mask,
        )

        loss_device_param = self._masked_mse(
            outputs["pred_device_param"],
            batch["target_device"].param_value,
            batch["target_device"].param_mask,
        )

        loss_device_spec = self._masked_mse(
            outputs["pred_device_spec"],
            batch["target_device"].spec_value,
            batch["target_device"].spec_mask,
        )

        loss_recon = loss_node + loss_edge + loss_device_param + loss_device_spec
        loss_state = loss_node + loss_edge + loss_device_param + loss_device_spec
        loss_total = self.alpha_recon * loss_recon + self.beta_state * loss_state

        self.log(f"{phase}_loss_total", loss_total, prog_bar=True, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_recon", loss_recon, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_state", loss_state, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_node", loss_node, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_edge", loss_edge, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_device_param", loss_device_param, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f"{phase}_loss_device_spec", loss_device_spec, sync_dist=True, batch_size=batch.num_graphs)

        return loss_total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }