from __future__ import annotations

import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict

class MultiTaskStateEstimationEngine(L.LightningModule):
    """
    Training engine for the hierarchical explicit-device architecture.

    Curriculum:
    - node/edge observability masking ramps up over epochs
    - device pseudo-measurement noise ramps up over epochs
    - spectrum-drop probability ramps up over epochs

    Logged losses:
    - total
    - recon
    - state
    - node
    - edge
    - device_param
    - device_spec

    #TODO: Split local autoencoding losses from graph-estimation losses more
    #      explicitly once a pretraining stage or separate local decoders exist.
    #TODO: Add device-type-aware loss masks so structurally irrelevant targets
    #      do not contribute to device losses even if padded tensors exist.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha_recon: float = 1.0,
        beta_state: float = 1.0,
        max_node_mask_ratio: float = 0.95,
        max_edge_mask_ratio: float = 0.95,
        max_device_noise_scale: float = 0.20,
        max_spectrum_drop_prob: float = 0.80,
        curriculum_epochs: int = 50,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha_recon = alpha_recon
        self.beta_state = beta_state

        self.max_node_mask_ratio = max_node_mask_ratio
        self.max_edge_mask_ratio = max_edge_mask_ratio
        self.max_device_noise_scale = max_device_noise_scale
        self.max_spectrum_drop_prob = max_spectrum_drop_prob
        self.curriculum_epochs = curriculum_epochs

        self.current_node_mask_ratio = 0.0
        self.current_edge_mask_ratio = 0.0
        self.current_device_noise_scale = 0.0
        self.current_spectrum_drop_prob = 0.0

        self.history = {
            "epoch": [],
            "train_loss_total": [],
            "val_loss_total": [],
            "train_loss_recon": [],
            "val_loss_recon": [],
            "train_loss_state": [],
            "val_loss_state": [],
            "train_loss_node": [],
            "val_loss_node": [],
            "train_loss_edge": [],
            "val_loss_edge": [],
            "train_loss_device_param": [],
            "val_loss_device_param": [],
            "train_loss_device_spec": [],
            "val_loss_device_spec": [],
            "node_mask_ratio": [],
            "edge_mask_ratio": [],
            "device_noise_scale": [],
            "spectrum_drop_prob": [],
        }

        self._epoch_acc = {}
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(
            batch,
            node_mask_ratio=self.current_node_mask_ratio,
            edge_mask_ratio=self.current_edge_mask_ratio,
            device_noise_scale=self.current_device_noise_scale,
            spectrum_drop_prob=self.current_spectrum_drop_prob,
        )

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0 or target.numel() == 0:
            device = pred.device if pred.numel() > 0 else target.device
            return torch.tensor(0.0, device=device)

        mask_f = mask.unsqueeze(-1).to(pred.dtype)
        diff_sq = ((pred - target) ** 2) * mask_f
        denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
        return diff_sq.sum() / denom

    def _start_phase_acc_if_needed(self, phase: str):
        if phase not in self._epoch_acc:
            self._epoch_acc[phase] = defaultdict(float)  # type: ignore[name-defined]
            self._epoch_acc[phase]["n"] = 0.0

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

        self._start_phase_acc_if_needed(phase)
        acc = self._epoch_acc[phase]
        acc["loss_total"] += float(loss_total.detach().cpu().item())
        acc["loss_recon"] += float(loss_recon.detach().cpu().item())
        acc["loss_state"] += float(loss_state.detach().cpu().item())
        acc["loss_node"] += float(loss_node.detach().cpu().item())
        acc["loss_edge"] += float(loss_edge.detach().cpu().item())
        acc["loss_device_param"] += float(loss_device_param.detach().cpu().item())
        acc["loss_device_spec"] += float(loss_device_spec.detach().cpu().item())
        acc["n"] += 1.0

        return loss_total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        saved_state = (
            self.current_node_mask_ratio,
            self.current_edge_mask_ratio,
            self.current_device_noise_scale,
            self.current_spectrum_drop_prob,
        )

        self.current_node_mask_ratio = self.max_node_mask_ratio
        self.current_edge_mask_ratio = self.max_edge_mask_ratio
        self.current_device_noise_scale = self.max_device_noise_scale
        self.current_spectrum_drop_prob = self.max_spectrum_drop_prob

        loss = self._shared_step(batch, "val")

        (
            self.current_node_mask_ratio,
            self.current_edge_mask_ratio,
            self.current_device_noise_scale,
            self.current_spectrum_drop_prob,
        ) = saved_state

        return loss

    def on_train_epoch_start(self):
        if self.current_epoch < self.curriculum_epochs:
            progress = self.current_epoch / float(self.curriculum_epochs)
        else:
            progress = 1.0

        self.current_node_mask_ratio = progress * self.max_node_mask_ratio
        self.current_edge_mask_ratio = progress * self.max_edge_mask_ratio
        self.current_device_noise_scale = progress * self.max_device_noise_scale
        self.current_spectrum_drop_prob = progress * self.max_spectrum_drop_prob

        self.log("node_mask_ratio", self.current_node_mask_ratio)
        self.log("edge_mask_ratio", self.current_edge_mask_ratio)
        self.log("device_noise_scale", self.current_device_noise_scale)
        self.log("spectrum_drop_prob", self.current_spectrum_drop_prob)

        self._epoch_acc = {}

    def on_train_epoch_end(self):
        self.history["epoch"].append(int(self.current_epoch))
        self.history["node_mask_ratio"].append(float(self.current_node_mask_ratio))
        self.history["edge_mask_ratio"].append(float(self.current_edge_mask_ratio))
        self.history["device_noise_scale"].append(float(self.current_device_noise_scale))
        self.history["spectrum_drop_prob"].append(float(self.current_spectrum_drop_prob))

        for phase in ["train", "val"]:
            acc = self._epoch_acc.get(phase, None)
            if acc is None or acc["n"] == 0:
                for metric in [
                    "loss_total",
                    "loss_recon",
                    "loss_state",
                    "loss_node",
                    "loss_edge",
                    "loss_device_param",
                    "loss_device_spec",
                ]:
                    self.history[f"{phase}_{metric}"].append(None)
                continue

            n = acc["n"]
            for metric in [
                "loss_total",
                "loss_recon",
                "loss_state",
                "loss_node",
                "loss_edge",
                "loss_device_param",
                "loss_device_spec",
            ]:
                self.history[f"{phase}_{metric}"].append(acc[metric] / n)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }