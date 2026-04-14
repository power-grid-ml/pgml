from __future__ import annotations

from pathlib import Path

import torch

from pgml.data_pipeline.step_dataloader import get_step_dataloader
from pgml.models.state_estimator import MultiModalStateEstimator
from pgml.config import config_dir, resource_dir, PipelineConfig

def debug_model_forward(base_dir: str | Path, dataset_ids: list[int]):
    loader = get_step_dataloader(
        base_data_dir=Path(base_dir),
        dataset_ids=dataset_ids,
        batch_size=1,
        num_workers=0,
    )

    batch = next(iter(loader))

    model = MultiModalStateEstimator(
        node_static_dim=batch["node"].static_x.shape[1],
        edge_static_dim=batch[("node", "physical", "node")].static_edge_attr.shape[1],
        device_static_dim=batch["device"].static_x.shape[1],
        node_value_dim=batch["node"].meas_value.shape[-1],
        edge_value_dim=batch["edge"].meas_value.shape[-1],
        device_param_value_dim=batch["device"].param_value.shape[-1] if batch["device"].param_value.ndim == 3 and batch["device"].param_value.shape[1] >= 0 else 1,
        device_spec_value_dim=batch["device"].spec_value.shape[-1] if batch["device"].spec_value.ndim == 3 and batch["device"].spec_value.shape[1] >= 0 else 6,
        hidden_dim=64,
    )

    with torch.no_grad():
        outputs = model(batch)

    print("=== Model Forward Debug ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")

if __name__ == "__main__":
    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)
    if Path(config.paths.input_dir).is_absolute():
        input_dir = Path(config.paths.input_dir)
    else:
        input_dir = Path(resource_dir) / config.paths.input_dir

    debug_model_forward(input_dir, [1,2,3])