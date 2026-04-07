from pathlib import Path
from torch_geometric.loader import DataLoader

from data_pipeline.topology import TopologyCache
from data_pipeline.scaler import DynamicStandardScaler
from data_pipeline.dataset import StreamingGridDataset


def get_train_dataloader(
        base_data_dir: Path,
        train_dataset_ids: list[int],
        stats_file: Path,
        batch_size: int = 32,
        num_workers: int = 4
) -> DataLoader:
    # Initialize Shared Components
    topo_cache = TopologyCache(base_data_dir)
    scaler = DynamicStandardScaler(stats_path=stats_file, table_name="node_data")

    # Gather physical dataset paths
    train_dirs = [base_data_dir / f"dataset_{did}" for did in train_dataset_ids]

    # Initialize Iterable Dataset
    dataset = StreamingGridDataset(
        dataset_dirs=train_dirs,
        topology_cache=topo_cache,
        scaler=scaler,
        batch_size_rows=200_000  # Tuned for RAM limits vs speed
    )

    # PyG DataLoader handles batching HeteroData automatically
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True  # Essential for fast GPU transfer
    )