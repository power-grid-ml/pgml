from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import lightning as L
import torch

from pgml.training.curriculum import TrainingCurriculum, TrainingStage, StageForwardConfig


class MultiTaskStateEstimationEngine(L.LightningModule):
    """
    Staged training engine for the hierarchical explicit-device architecture.

    The curriculum is the single source of truth for:
    - masking / noising / bypass_gnn
    - optimizer group learning rates
    - optimizer group weight decays
    - trainability / freezing
    - validation forward conditions
    - loss weights alpha_recon / beta_state

    Optimizer groups:
    - encoder
    - fusion
    - gnn
    - decoder
    - edge_static_encoder
    """

    def __init__(
        self,
        model: torch.nn.Module,
        curriculum: TrainingCurriculum,
    ):
        super().__init__()
        self.model = model
        self.curriculum = curriculum

        self.current_stage: TrainingStage = self.curriculum.get_stage(0)
        self.current_train_forward: StageForwardConfig = self.current_stage.train_forward
        self.current_val_forward: StageForwardConfig = self.current_stage.get_val_forward()

        self.history = {
            "epoch": [],
            "stage_name": [],
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
            "train_node_mask_ratio": [],
            "train_edge_mask_ratio": [],
            "train_device_noise_scale": [],
            "train_spectrum_drop_prob": [],
            "train_bypass_gnn": [],
            "val_node_mask_ratio": [],
            "val_edge_mask_ratio": [],
            "val_device_noise_scale": [],
            "val_spectrum_drop_prob": [],
            "val_bypass_gnn": [],
            "lr_encoder": [],
            "lr_fusion": [],
            "lr_gnn": [],
            "lr_decoder": [],
            "lr_edge_static_encoder": [],
        }

        self._epoch_acc = {}
        self._optimizer_group_name_to_index: Dict[str, int] = {}

        self.save_hyperparameters(ignore=["model", "curriculum"])

    def _get_named_module_groups(self) -> Dict[str, Iterable[torch.nn.Parameter]]:
        """
        Logical optimizer groups for staged training control.
        """
        groups = {
            "encoder": list(self.model.node_encoder.parameters())
                       + list(self.model.edge_encoder.parameters())
                       + list(self.model.device_encoder.parameters())
                       + list(self.model.node_masker.parameters())
                       + list(self.model.edge_masker.parameters())
                       + list(self.model.device_noiser.parameters()),
            "fusion": list(self.model.node_device_fusion.parameters()),
            "gnn": list(self.model.graph_estimator.parameters()),
            "decoder": list(self.model.node_decoder.parameters())
                       + list(self.model.edge_decoder.parameters())
                       + list(self.model.device_decoder.parameters()),
            "edge_static_encoder": list(self.model.edge_static_encoder.parameters()) if self.model.edge_static_encoder is not None else [],
        }
        return groups

    def _get_stage_group_cfgs(self, stage: TrainingStage):
        return {
            "encoder": stage.encoder,
            "fusion": stage.fusion,
            "gnn": stage.gnn,
            "decoder": stage.decoder,
            "edge_static_encoder": stage.edge_static_encoder,
        }

    def _apply_stage_trainability(self, stage: TrainingStage):
        groups = self._get_named_module_groups()
        cfgs = self._get_stage_group_cfgs(stage)

        for group_name, params in groups.items():
            trainable = cfgs[group_name].trainable
            for p in params:
                p.requires_grad = trainable

    def _apply_stage_optimizer_settings(self, stage: TrainingStage):
        opt = self.optimizers()
        if opt is None:
            return

        cfgs = self._get_stage_group_cfgs(stage)
        for group_name, group_idx in self._optimizer_group_name_to_index.items():
            cfg = cfgs[group_name]
            opt.param_groups[group_idx]["lr"] = float(cfg.lr)
            opt.param_groups[group_idx]["weight_decay"] = float(cfg.weight_decay)

    def _activate_stage(self, stage: TrainingStage):
        self.current_stage = stage
        self.current_train_forward = stage.train_forward
        self.current_val_forward = stage.get_val_forward()

        self._apply_stage_trainability(stage)
        self._apply_stage_optimizer_settings(stage)

        self.log("stage_bypass_gnn", float(stage.train_forward.bypass_gnn))
        self.log("train_node_mask_ratio", stage.train_forward.node_mask_ratio)
        self.log("train_edge_mask_ratio", stage.train_forward.edge_mask_ratio)
        self.log("train_device_noise_scale", stage.train_forward.device_noise_scale)
        self.log("train_spectrum_drop_prob", stage.train_forward.spectrum_drop_prob)

    def _forward_with_cfg(self, batch, cfg: StageForwardConfig):
        return self.model(
            batch,
            node_mask_ratio=float(cfg.node_mask_ratio),
            edge_mask_ratio=float(cfg.edge_mask_ratio),
            device_noise_scale=float(cfg.device_noise_scale),
            spectrum_drop_prob=float(cfg.spectrum_drop_prob),
            bypass_gnn=bool(cfg.bypass_gnn),
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
            self._epoch_acc[phase] = defaultdict(float)
            self._epoch_acc[phase]["n"] = 0.0

    def _shared_step(self, batch, phase: str):
        cfg = self.current_train_forward if phase == "train" else self.current_val_forward
        outputs = self._forward_with_cfg(batch, cfg)

        edge_type = ("node", "physical", "node")

        loss_node = self._masked_mse(
            outputs["pred_node_value"],
            batch["node"].target_voltage_value,
            batch["node"].target_voltage_mask,
        )

        loss_edge = self._masked_mse(
            outputs["pred_edge_value"],
            batch[edge_type].target_current_value,
            batch[edge_type].target_current_mask,
        )

        loss_device_param = self._masked_mse(
            outputs["pred_device_param"],
            batch["device"].target_param_value,
            batch["device"].target_param_mask,
        )

        loss_device_spec = self._masked_mse(
            outputs["pred_device_spec"],
            batch["device"].target_spec_value,
            batch["device"].target_spec_mask,
        )

        loss_recon = loss_node + loss_edge + loss_device_param + loss_device_spec

        # Still a proxy for now.
        loss_state = loss_node + loss_edge + loss_device_param + loss_device_spec

        loss_total = self.current_stage.alpha_recon * loss_recon + self.current_stage.beta_state * loss_state

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

    def forward(self, batch):
        return self._forward_with_cfg(batch, self.current_train_forward)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_train_epoch_start(self):
        stage = self.curriculum.get_stage(self.current_epoch)
        self._activate_stage(stage)
        self._epoch_acc = {}

    def on_train_epoch_end(self):
        self.history["epoch"].append(int(self.current_epoch))
        self.history["stage_name"].append(self.current_stage.name)

        self.history["train_node_mask_ratio"].append(float(self.current_train_forward.node_mask_ratio))
        self.history["train_edge_mask_ratio"].append(float(self.current_train_forward.edge_mask_ratio))
        self.history["train_device_noise_scale"].append(float(self.current_train_forward.device_noise_scale))
        self.history["train_spectrum_drop_prob"].append(float(self.current_train_forward.spectrum_drop_prob))
        self.history["train_bypass_gnn"].append(float(self.current_train_forward.bypass_gnn))

        self.history["val_node_mask_ratio"].append(float(self.current_val_forward.node_mask_ratio))
        self.history["val_edge_mask_ratio"].append(float(self.current_val_forward.edge_mask_ratio))
        self.history["val_device_noise_scale"].append(float(self.current_val_forward.device_noise_scale))
        self.history["val_spectrum_drop_prob"].append(float(self.current_val_forward.spectrum_drop_prob))
        self.history["val_bypass_gnn"].append(float(self.current_val_forward.bypass_gnn))

        self.history["lr_encoder"].append(float(self.current_stage.encoder.lr))
        self.history["lr_fusion"].append(float(self.current_stage.fusion.lr))
        self.history["lr_gnn"].append(float(self.current_stage.gnn.lr))
        self.history["lr_decoder"].append(float(self.current_stage.decoder.lr))
        self.history["lr_edge_static_encoder"].append(float(self.current_stage.edge_static_encoder.lr))

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
        groups = self._get_named_module_groups()
        cfgs = self._get_stage_group_cfgs(self.current_stage)

        param_groups = []
        group_name_to_index = {}

        for idx, (group_name, params) in enumerate(groups.items()):
            params = list(params)
            group_name_to_index[group_name] = idx

            cfg = cfgs[group_name]
            for p in params:
                p.requires_grad = cfg.trainable

            param_groups.append({
                "params": params,
                "lr": float(cfg.lr),
                "weight_decay": float(cfg.weight_decay),
                "name": group_name,
            })

        self._optimizer_group_name_to_index = group_name_to_index
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer