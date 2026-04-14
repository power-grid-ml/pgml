from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import torch


EDGE_TYPE = ("node", "physical", "node")


def _safe_mean(total: float, count: float) -> float:
    return total / count if count > 0 else 0.0


def _masked_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """
    Returns:
    - sum_squared_error
    - count_of_scalar_elements
    """
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
    """
    pred/target: [N, T, D]
    mask: [N, T]
    frequency: [N, T]
    """
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

    Report sections:
    - overall losses
    - encoder/decoder vs state estimator loss
    - split by entity type
    - split by frequency
    - split by device type

    Important current limitation:
    The current architecture logs reconstruction and state-estimator losses with
    identical formulas because there is not yet a separate local pretraining path.

    #TODO: Once explicit local decoders/pretraining are introduced, compute true
    #      encoder/decoder loss separately from graph-conditioned state loss.
    #TODO: Add static-feature-specific evaluation once static-feature prediction
    #      tasks are part of the supervised targets.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device

    def evaluate(self, engine, dataloader, output_dir: Path) -> str:
        engine.eval()

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
                    node_mask_ratio=engine.max_node_mask_ratio,
                    edge_mask_ratio=engine.max_edge_mask_ratio,
                    device_noise_scale=engine.max_device_noise_scale,
                    spectrum_drop_prob=engine.max_spectrum_drop_prob,
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

        summary_text = "\n".join(lines)
        return summary_text


class LossHistoryPlotter:
    """
    Creates matplotlib plots from the engine's in-memory history.

    #TODO: Add optional smoothing and per-step plotting if epoch-level curves
    #      are not sufficient for diagnosing training behavior.
    """

    def plot_from_engine(self, engine, output_path: Path, title: str = "Loss Curves"):
        history = getattr(engine, "history", None)
        if not history:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        epochs = history.get("epoch", [])

        base_metrics = [
            "loss_total", "loss_recon", "loss_state",
            "loss_node", "loss_edge", "loss_device_param", "loss_device_spec"
        ]

        # Use a distinct color cycle
        colors = plt.cm.tab10.colors
        ax = axes[0]

        for i, base_m in enumerate(base_metrics):
            train_k = f"train_{base_m}"
            val_k = f"val_{base_m}"
            color = colors[i % len(colors)]

            if train_k in history and len(history[train_k]) == len(epochs):
                ax.plot(epochs, history[train_k], label=train_k, color=color, linestyle="-")
            if val_k in history and len(history[val_k]) == len(epochs):
                ax.plot(epochs, history[val_k], label=val_k, color=color, linestyle="--")

        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot schedule
        ax2 = axes[1]
        sched_keys = ["node_mask_ratio", "edge_mask_ratio", "device_noise_scale", "spectrum_drop_prob"]
        for key in sched_keys:
            if key in history and len(history[key]) == len(epochs):
                ax2.plot(epochs, history[key], label=key)

        ax2.set_title("Curriculum / Masking Schedule")
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