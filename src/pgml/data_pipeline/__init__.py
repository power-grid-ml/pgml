from pathlib import Path
from torch_geometric.loader import DataLoader

from pgml.data_pipeline.topology import TopologyCache
from pgml.data_pipeline.scaler import BaseTorchScaler
from pgml.data_pipeline.dataset import StreamingGridDataset


def get_dataloader(
        base_data_dir: Path,
        dataset_ids: list[int],
        scaler: BaseTorchScaler,  # Pass the instantiated scaler here
        feature_prefixes: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        chunk_size_rows: int = 200_000,
) -> DataLoader:
    topo_cache = TopologyCache(base_data_dir)
    dataset_dirs = [base_data_dir / f"dataset_{did}" for did in dataset_ids]

    dataset = StreamingGridDataset(
        dataset_dirs=dataset_dirs,
        topology_cache=topo_cache,
        scaler=scaler,
        feature_prefixes=feature_prefixes,
        batch_size_rows=chunk_size_rows
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )