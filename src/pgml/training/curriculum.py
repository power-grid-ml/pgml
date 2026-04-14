# training/curriculum.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class OptimizerGroupConfig:
    lr: float
    weight_decay: float = 0.0
    trainable: bool = True


@dataclass
class StageForwardConfig:
    node_mask_ratio: float = 0.0
    edge_mask_ratio: float = 0.0
    device_noise_scale: float = 0.0
    spectrum_drop_prob: float = 0.0
    bypass_gnn: bool = False


@dataclass
class TrainingStage:
    """
    One stage of training.

    The stage becomes active from start_epoch onward until replaced by the next stage.
    """
    name: str
    start_epoch: int

    train_forward: StageForwardConfig
    val_forward: Optional[StageForwardConfig] = None

    encoder: OptimizerGroupConfig = field(
        default_factory=lambda: OptimizerGroupConfig(lr=1e-3, weight_decay=1e-4, trainable=True)
    )
    fusion: OptimizerGroupConfig = field(
        default_factory=lambda: OptimizerGroupConfig(lr=1e-3, weight_decay=1e-4, trainable=True)
    )
    gnn: OptimizerGroupConfig = field(
        default_factory=lambda: OptimizerGroupConfig(lr=1e-3, weight_decay=1e-4, trainable=True)
    )
    decoder: OptimizerGroupConfig = field(
        default_factory=lambda: OptimizerGroupConfig(lr=1e-3, weight_decay=1e-4, trainable=True)
    )
    edge_static_encoder: OptimizerGroupConfig = field(
        default_factory=lambda: OptimizerGroupConfig(lr=1e-3, weight_decay=1e-4, trainable=True)
    )

    alpha_recon: float = 1.0
    beta_state: float = 1.0

    def get_val_forward(self) -> StageForwardConfig:
        return self.val_forward if self.val_forward is not None else self.train_forward


class TrainingCurriculum:
    """
    Piecewise-constant stage schedule.
    """

    def __init__(self, stages: List[TrainingStage]):
        if not stages:
            raise ValueError("TrainingCurriculum requires at least one stage.")
        self.stages = sorted(stages, key=lambda s: s.start_epoch)

    def get_stage(self, epoch: int) -> TrainingStage:
        active = self.stages[0]
        for stage in self.stages:
            if epoch >= stage.start_epoch:
                active = stage
            else:
                break
        return active