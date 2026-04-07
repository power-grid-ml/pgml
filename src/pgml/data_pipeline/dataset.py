import math
from pathlib import Path
from typing import List, Iterator, Optional

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import HeteroData

from data_pipeline.topology import TopologyCache
from data_pipeline.scaler import BaseTorchScaler


class StreamingGridDataset(IterableDataset):
    def __init__(
            self,
            dataset_dirs: List[Path],
            topology_cache: TopologyCache,
            scaler: BaseTorchScaler,
            batch_size_rows: int = 200_000,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.topology_cache = topology_cache
        self.scaler = scaler
        self.batch_size_rows = batch_size_rows

    def polar_to_rect(self, magnitudes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Converts polar coordinates to interleaved rectangular coordinates (real, imag)."""
        real_parts = magnitudes * torch.cos(angles)
        imag_parts = magnitudes * torch.sin(angles)

        # Interleave:[v1_real, v1_imag, v2_real, v2_imag, ...]
        rect_tensor = torch.empty((magnitudes.shape[0], magnitudes.shape[1] * 2), dtype=torch.float32)
        rect_tensor[:, 0::2] = real_parts
        rect_tensor[:, 1::2] = imag_parts
        return rect_tensor

    def _process_parquet_file(self, file_path: Path, topo_base: HeteroData, frequency: float) -> Iterator[HeteroData]:
        parquet_file = pq.ParquetFile(file_path)

        # Buffer for steps that get split across pyarrow batches
        step_buffer = {'steps': [], 'data': []}

        for batch in parquet_file.iter_batches(batch_size=self.batch_size_rows):
            # 1. Convert Arrow to PyTorch efficiently (zero-copy where possible)
            step_col = torch.from_numpy(batch.column("step").to_numpy())

            # Extract Polar components
            mags = torch.stack([
                torch.from_numpy(batch.column("v1").to_numpy()),
                torch.from_numpy(batch.column("v2").to_numpy()),
                torch.from_numpy(batch.column("v3").to_numpy())
            ], dim=1)

            angs = torch.stack([
                torch.from_numpy(batch.column("v1_angle").to_numpy()),
                torch.from_numpy(batch.column("v2_angle").to_numpy()),
                torch.from_numpy(batch.column("v3_angle").to_numpy())
            ], dim=1)

            # 2. Polar to Rectangular
            rect_features = self.polar_to_rect(mags, angs)

            # 3. Apply Scaling
            scaled_features = self.scaler.fit_transform(rect_features)

            # 4. Find boundaries of steps (vectorized)
            # unique_consecutive returns the unique elements, their inverse indices, and counts
            unique_steps, counts = torch.unique_consecutive(step_col, return_counts=True)
            split_indices = torch.cumsum(counts, dim=0)[:-1]

            step_tensors = torch.tensor_split(scaled_features, split_indices)
            step_ids = unique_steps.tolist()

            # 5. Yield graphs, handling boundary overlap
            for i, (s_id, s_tensor) in enumerate(zip(step_ids, step_tensors)):

                # If this step matches the buffered step, append it
                if step_buffer['steps'] and s_id == step_buffer['steps'][-1]:
                    step_buffer['data'][-1] = torch.cat([step_buffer['data'][-1], s_tensor], dim=0)
                else:
                    # New step encountered. Yield the previous complete step in the buffer
                    if step_buffer['steps']:
                        yield self._build_graph(topo_base, step_buffer['data'].pop(0), frequency)
                        step_buffer['steps'].pop(0)

                    # Add current step to buffer
                    step_buffer['steps'].append(s_id)
                    step_buffer['data'].append(s_tensor)

        # Yield the final step left in the buffer after file finishes
        if step_buffer['steps']:
            yield self._build_graph(topo_base, step_buffer['data'].pop(0), frequency)

    def _build_graph(self, topo_base: HeteroData, dynamic_node_x: torch.Tensor, frequency: float) -> HeteroData:
        """Fuses static topology with dynamic states for a single step."""
        graph = topo_base.clone()

        # Combine static node features with dynamic scaled node states
        graph['node'].x = torch.cat([graph['node'].static_x, dynamic_node_x], dim=-1)

        # Embed physical frequency as a graph-level feature
        graph.frequency = torch.tensor([frequency], dtype=torch.float32)
        return graph

    def __iter__(self):
        # Handle PyTorch multiprocessing via DataLoader workers
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process data loading
            process_dirs = self.dataset_dirs
        else:
            # Split dataset directories among workers to prevent duplication
            per_worker = int(math.ceil(len(self.dataset_dirs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset_dirs))
            process_dirs = self.dataset_dirs[start:end]

        for d_dir in process_dirs:
            # Read metadata
            import json
            with open(d_dir / "metadata.json", "r") as f:
                meta = json.load(f)

            topo_id = meta["topology_id"]
            topo_base = self.topology_cache.get_topology(topo_id)

            # frequency is assumed to be in dataset metadata or static per dataset.
            # E.g., if set statically per dataset:
            frequency = float(meta["dataset"].get("frequency", 50.0))

            # Process node_data (can be extended to zip/join with edge_data)
            node_file = d_dir / "node_data" / "data.parquet"
            if node_file.exists():
                yield from self._process_parquet_file(node_file, topo_base, frequency)