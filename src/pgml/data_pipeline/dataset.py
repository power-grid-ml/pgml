import math
from pathlib import Path
from typing import List, Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import HeteroData

from pgml.data_pipeline.topology import TopologyCache
from pgml.data_pipeline.scaler import BaseTorchScaler


class StreamingGridDataset(IterableDataset):
    def __init__(
            self,
            dataset_dirs: List[Path],
            topology_cache: TopologyCache,
            scaler: BaseTorchScaler,
            feature_prefixes: List[str],  # E.g.,["v1", "v2", "v3"]
            batch_size_rows: int = 200_000,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.topology_cache = topology_cache
        self.scaler = scaler
        self.feature_prefixes = feature_prefixes
        self.batch_size_rows = batch_size_rows

    def polar_to_rect(self, magnitudes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Converts polar coordinates to interleaved rectangular coordinates (real, imag)."""
        real_parts = magnitudes * torch.cos(angles)
        imag_parts = magnitudes * torch.sin(angles)

        # Interleave: [v1_real, v1_imag, v2_real, v2_imag, ...]
        rect_tensor = torch.empty((magnitudes.shape[0], magnitudes.shape[1] * 2), dtype=torch.float32)
        rect_tensor[:, 0::2] = real_parts
        rect_tensor[:, 1::2] = imag_parts
        return rect_tensor

    def _yield_graphs_for_step(
            self,
            topo_base: HeteroData,
            rect_data: torch.Tensor,
            freqs: torch.Tensor,
            node_ids: torch.Tensor
    ) -> Iterator[HeteroData]:
        """
        Takes all rows for a single time step, groups them by frequency,
        sorts them by node_id, scales them, and yields discrete physical graphs.
        """
        unique_freqs = torch.unique(freqs)

        for freq in unique_freqs:
            # 1. Mask rows for this specific frequency
            mask = (freqs == freq)
            f_data = rect_data[mask]
            f_node_ids = node_ids[mask]

            # 2. Sort by node_id to perfectly align with static_x
            sort_idx = torch.argsort(f_node_ids)
            sorted_data = f_data[sort_idx]

            # 3. Apply Scaling
            freq_val = float(freq.item())
            freq_str = str(int(freq_val))  # Matches the str(int()) logic in stats_compiler
            scaled_data = self.scaler(sorted_data, group_key=freq_str)

            # 4. Construct Graph
            graph = topo_base.clone()

            # dim=-1 concatenates columns.
            # Because of sorting, row N dynamically aligns with row N statically.
            graph['node'].x = torch.cat([graph['node'].static_x, scaled_data], dim=-1)
            graph.frequency = torch.tensor([freq_val], dtype=torch.float32)

            yield graph

    def _process_parquet_file(self, file_path: Path, topo_base: HeteroData) -> Iterator[HeteroData]:
        parquet_file = pq.ParquetFile(file_path)

        # Buffer tracks fragments of a step if it gets split across two pyarrow batches
        step_buffer = {'step_id': None, 'data': [], 'node_ids': [], 'freqs': []}

        for batch in parquet_file.iter_batches(batch_size=self.batch_size_rows):
            # .copy() prevents PyTorch non-writable memory warnings from PyArrow mappings
            step_col = torch.from_numpy(batch.column("step").to_numpy().copy())
            freq_col = torch.from_numpy(batch.column("frequency").to_numpy().copy())
            node_id_col = torch.from_numpy(batch.column("node_id").to_numpy().copy())

            # Dynamically fetch polar data based on configuration
            mags = []
            angs = []
            for p in self.feature_prefixes:
                mags.append(torch.from_numpy(batch.column(p).to_numpy().copy()))
                angs.append(torch.from_numpy(batch.column(f"{p}_angle").to_numpy().copy()))

            mags = torch.stack(mags, dim=1)
            angs = torch.stack(angs, dim=1)
            rect_features = self.polar_to_rect(mags, angs)

            # Vectorized chunking by step
            unique_steps, counts = torch.unique_consecutive(step_col, return_counts=True)
            split_indices = torch.cumsum(counts, dim=0)[:-1]

            step_tensors = torch.tensor_split(rect_features, split_indices)
            step_node_ids = torch.tensor_split(node_id_col, split_indices)
            step_freqs = torch.tensor_split(freq_col, split_indices)
            step_ids = unique_steps.tolist()

            # Iterate through the chunks in this batch
            for s_id, s_data, s_n_ids, s_freqs in zip(step_ids, step_tensors, step_node_ids, step_freqs):
                if step_buffer['step_id'] is None:
                    step_buffer['step_id'] = s_id

                if s_id == step_buffer['step_id']:
                    # Accumulate parts of the same step
                    step_buffer['data'].append(s_data)
                    step_buffer['node_ids'].append(s_n_ids)
                    step_buffer['freqs'].append(s_freqs)
                else:
                    # Step changed! Yield the fully buffered step
                    cat_data = torch.cat(step_buffer['data'], dim=0)
                    cat_n_ids = torch.cat(step_buffer['node_ids'], dim=0)
                    cat_freqs = torch.cat(step_buffer['freqs'], dim=0)

                    yield from self._yield_graphs_for_step(topo_base, cat_data, cat_freqs, cat_n_ids)

                    # Reset buffer for the new step
                    step_buffer['step_id'] = s_id
                    step_buffer['data'] = [s_data]
                    step_buffer['node_ids'] = [s_n_ids]
                    step_buffer['freqs'] = [s_freqs]

        # File exhausted. Yield whatever remains in the buffer.
        if step_buffer['step_id'] is not None:
            cat_data = torch.cat(step_buffer['data'], dim=0)
            cat_n_ids = torch.cat(step_buffer['node_ids'], dim=0)
            cat_freqs = torch.cat(step_buffer['freqs'], dim=0)
            yield from self._yield_graphs_for_step(topo_base, cat_data, cat_freqs, cat_n_ids)

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            process_dirs = self.dataset_dirs
        else:
            per_worker = int(math.ceil(len(self.dataset_dirs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset_dirs))
            process_dirs = self.dataset_dirs[start:end]

        for d_dir in process_dirs:
            import json
            with open(d_dir / "metadata.json", "r") as f:
                meta = json.load(f)

            topo_id = meta["topology_id"]
            topo_base = self.topology_cache.get_topology(topo_id)

            node_file = d_dir / "node_data" / "data.parquet"
            if node_file.exists():
                # Frequency is no longer static. Passed directly to processor.
                yield from self._process_parquet_file(node_file, topo_base)