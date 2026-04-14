from __future__ import annotations

from pathlib import Path

from pgml.data_pipeline.tokenizer import MeasurementTokenizer
from pgml.data_pipeline.step_dataset import StreamingStepDataset
from pgml.data_pipeline.topology import TopologyCache
from pgml.config import config_dir, resource_dir, PipelineConfig

def debug_first_step(base_dir: Path, dataset_id: int):
    base_dir = Path(base_dir)
    topology_cache = TopologyCache(base_dir)
    tokenizer = MeasurementTokenizer()

    dataset = StreamingStepDataset(
        dataset_dirs=[base_dir / f"dataset_{dataset_id}"],
        topology_cache=topology_cache,
        tokenizer=tokenizer,
    )

    sample = next(iter(dataset))

    print("=== Full Step Graph Debug ===")
    print("dataset_id:", sample.dataset_id)
    print("step:", sample.step)
    print("topology_id:", sample.topology_id)

    print("node static:", sample["node"].static_x.shape)
    print("node meas value:", sample["node"].meas_value.shape)
    print("node meas freq:", sample["node"].meas_frequency.shape)
    print("node meas mask:", sample["node"].meas_mask.shape)

    print("edge static:", sample[("node", "physical", "node")].static_edge_attr.shape)
    print("edge meas value:", sample["edge"].meas_value.shape)
    print("edge meas freq:", sample["edge"].meas_frequency.shape)
    print("edge meas mask:", sample["edge"].meas_mask.shape)

    print("device static:", sample["device"].static_x.shape)
    print("device node_index:", sample["device"].node_index.shape)
    print("device param value:", sample["device"].param_value.shape)
    print("device spec value:", sample["device"].spec_value.shape)

    print("target node voltage:", sample["target_node"].voltage_value.shape)
    print("target edge current:", sample["target_edge"].current_value.shape)
    print("target device param:", sample["target_device"].param_value.shape)
    print("target device spec:", sample["target_device"].spec_value.shape)

if __name__ == "__main__":
    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)
    if Path(config.paths.input_dir).is_absolute():
        input_dir = Path(config.paths.input_dir)
    else:
        input_dir = Path(resource_dir) / config.paths.input_dir
    debug_first_step(input_dir, 1)
    debug_first_step(input_dir, 2)
    debug_first_step(input_dir, 3)