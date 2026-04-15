from __future__ import annotations

from pathlib import Path
from typing import List

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from pgml.data_pipeline.tokenizer import MeasurementTokenizer
from pgml.data_pipeline.dataset import StreamingDataset
from pgml.data_pipeline.topology import TopologyCache


def get_dataloader(
    base_data_dir: Path,
    dataset_ids: List[int],
    batch_size: int = 1,
    num_workers: int = 0,
    node_feature_prefixes: tuple[str, ...] = ("v1", "v2", "v3"),
    edge_current_prefixes: tuple[str, ...] = ("i1", "i2", "i3"),
    edge_power_prefixes: tuple[str, ...] = (),
    spectrum_prefixes: tuple[str, ...] = ("spectrum1", "spectrum2", "spectrum3"),
):
    dataset_dirs = [base_data_dir / f"dataset_{d_id}" for d_id in dataset_ids]
    topology_cache = TopologyCache(base_data_dir)
    tokenizer = MeasurementTokenizer()

    dataset = StreamingDataset(
        dataset_dirs=dataset_dirs,
        topology_cache=topology_cache,
        tokenizer=tokenizer,
        node_feature_prefixes=node_feature_prefixes,
        edge_current_prefixes=edge_current_prefixes,
        edge_power_prefixes=edge_power_prefixes,
        spectrum_prefixes=spectrum_prefixes,
    )

    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )