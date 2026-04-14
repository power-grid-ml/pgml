from __future__ import annotations

import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import defaultdict

class StepWiseCurriculum:
    """Helper to define custom step-wise schedules."""

    def __init__(self, schedule: dict[int, dict[str, float]]):
        self.schedule = schedule
        self.sorted_epochs = sorted(schedule.keys())

    def __call__(self, epoch: int) -> dict[str, float]:
        active_key = self.sorted_epochs[0]
        for e in self.sorted_epochs:
            if epoch >= e:
                active_key = e
        return self.schedule[active_key]

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
            curriculum_fn: Callable[[int], Dict[str, float]],
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            alpha_recon: float = 1.0,
            beta_state: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.curriculum_fn = curriculum_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha_recon = alpha_recon
        self.beta_state = beta_state

        self.current_masking_params = {
            "node_mask_ratio": 0.0,
            "edge_mask_ratio": 0.0,
            "device_noise_scale": 0.0,
            "spectrum_drop_prob": 0.0,
        }

        self.history = {
            "epoch": [],
            "train_loss_total": [], "val_loss_total": [],
            "train_loss_recon": [], "val_loss_recon": [],
            "train_loss_state": [], "val_loss_state": [],
            "train_loss_node": [], "val_loss_node": [],
            "train_loss_edge": [], "val_loss_edge": [],
            "train_loss_device_param": [], "val_loss_device_param": [],
            "train_loss_device_spec": [], "val_loss_device_spec": [],
            "node_mask_ratio": [], "edge_mask_ratio": [],
            "device_noise_scale": [], "spectrum_drop_prob": [],
        }

        self._epoch_acc = {}
        # Ignore complex objects for hparams logging
        self.save_hyperparameters(ignore=["model", "curriculum_fn"])

    def forward(self, batch):
        return self.model(
            batch,
            **self.current_masking_params
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
        # Force eval phase to max difficulty for validation, or specific test params
        saved_params = self.current_masking_params.copy()

        # Optionally, you can pass a specific validation config here.
        # For now, we will evaluate using the current epoch's curriculum difficulty.
        loss = self._shared_step(batch, "val")

        self.current_masking_params = saved_params
        return loss

    def on_train_epoch_start(self):
        new_params = self.curriculum_fn(self.current_epoch)
        self.current_masking_params.update(new_params)

        for k, v in self.current_masking_params.items():
            self.log(k, v)

        self._epoch_acc = {}

    def on_train_epoch_end(self):
        self.history["epoch"].append(int(self.current_epoch))
        for k, v in self.current_masking_params.items():
            self.history[k].append(float(v))

        for phase in ["train", "val"]:
            acc = self._epoch_acc.get(phase, None)
            if acc is None or acc["n"] == 0:
                for metric in ["loss_total", "loss_recon", "loss_state", "loss_node", "loss_edge",
                               "loss_device_param", "loss_device_spec"]:
                    self.history[f"{phase}_{metric}"].append(None)
                continue

            n = acc["n"]
            for metric in ["loss_total", "loss_recon", "loss_state", "loss_node", "loss_edge", "loss_device_param",
                           "loss_device_spec"]:
                self.history[f"{phase}_{metric}"].append(acc[metric] / n)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }