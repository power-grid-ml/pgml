from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, List

from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import HeteroData

from pgml.data_pipeline.tokenizer import MeasurementTokenizer
from pgml.data_pipeline.graph_assembler import GraphAssembler
from pgml.data_pipeline.topology import TopologyCache


class StreamingDataset(IterableDataset):
    """
    Streams one full graph per simulation step.

    Each yielded item is a HeteroData object containing:
    - static topology
    - node measurement tokens
    - edge measurement tokens
    - explicit device features and dynamic tokens
    - corresponding targets

    This replaces the older design that yielded one graph per (step, frequency).
    """

    def __init__(
        self,
        dataset_dirs: List[Path],
        topology_cache: TopologyCache,
        tokenizer: MeasurementTokenizer | None = None,
        node_feature_prefixes: tuple[str, ...] = ("v1", "v2", "v3"),
        edge_current_prefixes: tuple[str, ...] = ("i1", "i2", "i3"),
        edge_power_prefixes: tuple[str, ...] = (),
        spectrum_prefixes: tuple[str, ...] = ("spectrum1", "spectrum2", "spectrum3"),
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.tokenizer = tokenizer or MeasurementTokenizer()
        self.assembler = GraphAssembler(
            topology_cache=topology_cache,
            tokenizer=self.tokenizer,
            node_feature_prefixes=node_feature_prefixes,
            edge_current_prefixes=edge_current_prefixes,
            edge_power_prefixes=edge_power_prefixes,
            spectrum_prefixes=spectrum_prefixes,
        )

    def __iter__(self) -> Iterator[HeteroData]:
        worker_info = get_worker_info()

        if worker_info is None:
            process_dirs = self.dataset_dirs
        else:
            per_worker = int(math.ceil(len(self.dataset_dirs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.dataset_dirs))
            process_dirs = self.dataset_dirs[start:end]

        for dataset_dir in process_dirs:
            steps = self.assembler.list_steps(dataset_dir)
            for step in steps:
                yield self.assembler.assemble_graph(dataset_dir=dataset_dir, step=step)