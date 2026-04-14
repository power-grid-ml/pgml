from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch

from pgml.training.curriculum import StageForwardConfig

EDGE_TYPE = ("node", "physical", "node")


def _safe_mean(total: float, count: float) -> float:
    return total / count if count > 0 else 0.0


def _masked_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    if pred.numel() == 0 or target.numel() == 0 or mask.numel() == 0:
        return 0.0, 0.0

    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    diff_sq = ((pred - target) ** 2) * mask_f
    count = float(mask_f.sum().item() * pred.shape[-1])
    return float(diff_sq.sum().item()), count


def _accumulate_by_frequency(
    accumulator: Dict[str, Dict[str, float]],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    frequency: torch.Tensor,
):
    if pred.numel() == 0:
        return

    n, t, d = pred.shape
    for i in range(n):
        for j in range(t):
            if not bool(mask[i, j].item()):
                continue
            freq = str(float(frequency[i, j].item()))
            se = float(((pred[i, j] - target[i, j]) ** 2).sum().item())
            accumulator[freq]["se"] += se
            accumulator[freq]["count"] += d


def _accumulate_by_device_type(
    accumulator: Dict[str, Dict[str, float]],
    pred_param: torch.Tensor,
    target_param: torch.Tensor,
    mask_param: torch.Tensor,
    pred_spec: torch.Tensor,
    target_spec: torch.Tensor,
    mask_spec: torch.Tensor,
    device_type: torch.Tensor,
):
    if device_type.numel() == 0:
        return

    name_map = {
        0: "load",
        1: "generator",
        2: "vsource",
        3: "injected",
    }

    for i in range(device_type.shape[0]):
        dtype = name_map.get(int(device_type[i].item()), f"unknown_{int(device_type[i].item())}")

        if pred_param.numel() > 0 and pred_param.shape[0] > i:
            se_p, count_p = _masked_squared_error(
                pred_param[i:i + 1],
                target_param[i:i + 1],
                mask_param[i:i + 1],
            )
            accumulator[dtype]["param_se"] += se_p
            accumulator[dtype]["param_count"] += count_p

        if pred_spec.numel() > 0 and pred_spec.shape[0] > i:
            se_s, count_s = _masked_squared_error(
                pred_spec[i:i + 1],
                target_spec[i:i + 1],
                mask_spec[i:i + 1],
            )
            accumulator[dtype]["spec_se"] += se_s
            accumulator[dtype]["spec_count"] += count_s


class ValidationEvaluator:
    """
    Computes text-based validation summaries for the full validation set.

    #TODO: Once explicit local decoders/pretraining are introduced, compute true
    #      encoder/decoder loss separately from graph-conditioned state loss.
    #TODO: Add static-feature-specific evaluation once static-feature prediction
    #      tasks are part of the supervised targets.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device

    def evaluate(
        self,
        engine,
        dataloader,
        output_dir: Path,
        forward_cfg: Optional[StageForwardConfig] = None,
    ) -> str:
        engine.eval()

        if forward_cfg is None:
            forward_cfg = engine.current_val_forward

        totals = defaultdict(float)

        freq_node = defaultdict(lambda: {"se": 0.0, "count": 0.0})
        freq_edge = defaultdict(lambda: {"se": 0.0, "count": 0.0})
        freq_device_spec = defaultdict(lambda: {"se": 0.0, "count": 0.0})
        device_type_stats = defaultdict(
            lambda: {
                "param_se": 0.0,
                "param_count": 0.0,
                "spec_se": 0.0,
                "spec_count": 0.0,
            }
        )

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(engine.device)
                outputs = engine.model(
                    batch,
                    node_mask_ratio=float(forward_cfg.node_mask_ratio),
                    edge_mask_ratio=float(forward_cfg.edge_mask_ratio),
                    device_noise_scale=float(forward_cfg.device_noise_scale),
                    spectrum_drop_prob=float(forward_cfg.spectrum_drop_prob),
                    bypass_gnn=bool(forward_cfg.bypass_gnn),
                )

                node_se, node_count = _masked_squared_error(
                    outputs["pred_node_value"],
                    batch["target_node"].voltage_value,
                    batch["target_node"].voltage_mask,
                )
                edge_se, edge_count = _masked_squared_error(
                    outputs["pred_edge_value"],
                    batch["target_edge"].current_value,
                    batch["target_edge"].current_mask,
                )
                dev_param_se, dev_param_count = _masked_squared_error(
                    outputs["pred_device_param"],
                    batch["target_device"].param_value,
                    batch["target_device"].param_mask,
                )
                dev_spec_se, dev_spec_count = _masked_squared_error(
                    outputs["pred_device_spec"],
                    batch["target_device"].spec_value,
                    batch["target_device"].spec_mask,
                )

                totals["node_se"] += node_se
                totals["node_count"] += node_count
                totals["edge_se"] += edge_se
                totals["edge_count"] += edge_count
                totals["dev_param_se"] += dev_param_se
                totals["dev_param_count"] += dev_param_count
                totals["dev_spec_se"] += dev_spec_se
                totals["dev_spec_count"] += dev_spec_count

                _accumulate_by_frequency(
                    freq_node,
                    outputs["pred_node_value"],
                    batch["target_node"].voltage_value,
                    batch["target_node"].voltage_mask,
                    batch["target_node"].voltage_frequency,
                )
                _accumulate_by_frequency(
                    freq_edge,
                    outputs["pred_edge_value"],
                    batch["target_edge"].current_value,
                    batch["target_edge"].current_mask,
                    batch["target_edge"].current_frequency,
                )
                _accumulate_by_frequency(
                    freq_device_spec,
                    outputs["pred_device_spec"],
                    batch["target_device"].spec_value,
                    batch["target_device"].spec_mask,
                    batch["target_device"].spec_frequency,
                )

                _accumulate_by_device_type(
                    device_type_stats,
                    outputs["pred_device_param"],
                    batch["target_device"].param_value,
                    batch["target_device"].param_mask,
                    outputs["pred_device_spec"],
                    batch["target_device"].spec_value,
                    batch["target_device"].spec_mask,
                    batch["device"].device_type,
                )

        node_mse = _safe_mean(totals["node_se"], totals["node_count"])
        edge_mse = _safe_mean(totals["edge_se"], totals["edge_count"])
        dev_param_mse = _safe_mean(totals["dev_param_se"], totals["dev_param_count"])
        dev_spec_mse = _safe_mean(totals["dev_spec_se"], totals["dev_spec_count"])

        loss_encoder_decoder = node_mse + edge_mse + dev_param_mse + dev_spec_mse
        loss_state_estimator = node_mse + edge_mse + dev_param_mse + dev_spec_mse

        lines = []
        lines.append("Validation Summary")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Validation forward configuration")
        lines.append("-" * 80)
        lines.append(f"node_mask_ratio: {forward_cfg.node_mask_ratio}")
        lines.append(f"edge_mask_ratio: {forward_cfg.edge_mask_ratio}")
        lines.append(f"device_noise_scale: {forward_cfg.device_noise_scale}")
        lines.append(f"spectrum_drop_prob: {forward_cfg.spectrum_drop_prob}")
        lines.append(f"bypass_gnn: {forward_cfg.bypass_gnn}")
        lines.append("")

        lines.append("Overall Losses")
        lines.append("-" * 80)
        lines.append(f"Encoder/Decoder Loss (current proxy): {loss_encoder_decoder:.8f}")
        lines.append(f"State Estimator Loss (current proxy): {loss_state_estimator:.8f}")
        lines.append("")
        lines.append("By dynamic target type")
        lines.append("-" * 80)
        lines.append(f"Node dynamic voltage MSE: {node_mse:.8f}")
        lines.append(f"Edge dynamic current/power token MSE: {edge_mse:.8f}")
        lines.append(f"Device dynamic parameter MSE: {dev_param_mse:.8f}")
        lines.append(f"Device dynamic spectrum MSE: {dev_spec_mse:.8f}")
        lines.append("")
        lines.append("By static target type")
        lines.append("-" * 80)
        lines.append("Static-target evaluation not yet applicable.")
        lines.append("Current model uses static features as conditioning inputs, not as supervised outputs.")
        lines.append("")

        lines.append("By node/edge/device category")
        lines.append("-" * 80)
        lines.append(f"node: {node_mse:.8f}")
        lines.append(f"edge: {edge_mse:.8f}")
        lines.append(f"device_param: {dev_param_mse:.8f}")
        lines.append(f"device_spec: {dev_spec_mse:.8f}")
        lines.append("")

        lines.append("By frequency: node targets")
        lines.append("-" * 80)
        for freq, stats in sorted(freq_node.items(), key=lambda x: float(x[0])):
            mse = _safe_mean(stats["se"], stats["count"])
            lines.append(f"{freq:>10s} Hz : {mse:.8f}")
        lines.append("")

        lines.append("By frequency: edge targets")
        lines.append("-" * 80)
        for freq, stats in sorted(freq_edge.items(), key=lambda x: float(x[0])):
            mse = _safe_mean(stats["se"], stats["count"])
            lines.append(f"{freq:>10s} Hz : {mse:.8f}")
        lines.append("")

        lines.append("By frequency: device spectrum targets")
        lines.append("-" * 80)
        for freq, stats in sorted(freq_device_spec.items(), key=lambda x: float(x[0])):
            mse = _safe_mean(stats["se"], stats["count"])
            lines.append(f"{freq:>10s} Hz : {mse:.8f}")
        lines.append("")

        lines.append("By device type")
        lines.append("-" * 80)
        for dtype, stats in sorted(device_type_stats.items(), key=lambda x: x[0]):
            param_mse = _safe_mean(stats["param_se"], stats["param_count"])
            spec_mse = _safe_mean(stats["spec_se"], stats["spec_count"])
            lines.append(f"{dtype:>12s} | param_mse={param_mse:.8f} | spec_mse={spec_mse:.8f}")

        return "\n".join(lines)


class LossHistoryPlotter:
    def plot_from_engine(self, engine, output_path: Path, title: str = "Loss Curves"):
        history = getattr(engine, "history", None)
        if not history:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        epochs = history.get("epoch", [])

        metric_pairs = [
            ("loss_total", "Total Loss"),
            ("loss_recon", "Recon Loss"),
            ("loss_state", "State Loss"),
            ("loss_node", "Node Loss"),
            ("loss_edge", "Edge Loss"),
            ("loss_device_param", "Device Param Loss"),
            ("loss_device_spec", "Device Spec Loss"),
        ]

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        ax = axes[0]
        for i, (metric_key, label) in enumerate(metric_pairs):
            color = colors[i % len(colors)]
            train_key = f"train_{metric_key}"
            val_key = f"val_{metric_key}"

            if train_key in history and len(history[train_key]) == len(epochs):
                ax.plot(epochs, history[train_key], linestyle="-", color=color, label=f"train_{label}")
            if val_key in history and len(history[val_key]) == len(epochs):
                ax.plot(epochs, history[val_key], linestyle="--", color=color, label=f"val_{label}")

        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        sched_keys = [
            "train_node_mask_ratio",
            "train_edge_mask_ratio",
            "train_device_noise_scale",
            "train_spectrum_drop_prob",
            "train_bypass_gnn",
            "lr_encoder",
            "lr_fusion",
            "lr_gnn",
            "lr_decoder",
            "lr_edge_static_encoder",
        ]
        for key in sched_keys:
            if key in history and len(history[key]) == len(epochs):
                ax2.plot(epochs, history[key], label=key)

        ax2.set_title("Curriculum / Training Schedule")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def export_training_history_json(engine, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = getattr(engine, "history", {})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)