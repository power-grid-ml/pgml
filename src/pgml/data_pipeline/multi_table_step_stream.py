from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import polars as pl

from pgml.data_pipeline.step_graph_stream import ParquetStepStream, StepChunk


@dataclass
class StepTableBundle:
    dataset_id: int
    topology_id: int
    step: int
    tables: Dict[str, pl.DataFrame]


class MultiTableStepStream:
    """
    Streams all relevant dynamic tables of one dataset and aligns them by step.

    The implementation keeps exactly one current chunk per table stream in memory,
    so memory usage is bounded by:
    - one buffered step per table
    - one chunk per table during iteration

    Missing tables are represented as empty DataFrames.
    """

    TABLE_SPECS = {
        "node_data": {
            "relative_file": "node_data/data.parquet",
            "columns": ["step", "node_id", "frequency", "v1", "v1_angle", "v2", "v2_angle", "v3", "v3_angle"],
        },
        "edge_data": {
            "relative_file": "edge_data/data.parquet",
            "columns": ["step", "edge_id", "frequency", "i1", "i1_angle", "i2", "i2_angle", "i3", "i3_angle"],
        },
        "load_parameters": {
            "relative_file": "load_parameters/data.parquet",
            "columns": ["step", "load_id", "p1", "q1", "p2", "q2", "p3", "q3"],
        },
        "generator_parameters": {
            "relative_file": "generator_parameters/data.parquet",
            "columns": ["step", "generator_id", "p1", "q1", "p2", "q2", "p3", "q3"],
        },
        "vsource_parameters": {
            "relative_file": "vsource_parameters/data.parquet",
            "columns": ["step", "vsource_id", "pu1", "pu2", "pu3"],
        },
        "injected_error_parameters": {
            "relative_file": "injected_error_parameters/data.parquet",
            "columns": ["step", "node_id", "sc1_mva"],
        },
        "spectrum": {
            "relative_file": "spectrum/data.parquet",
            "columns": [
                "step", "parent_type", "parent_id", "frequency",
                "spectrum1", "spectrum1_angle",
                "spectrum2", "spectrum2_angle",
                "spectrum3", "spectrum3_angle",
            ],
        },
    }

    def __init__(
        self,
        dataset_dir: Path,
        chunk_size_rows: int = 50_000,
    ):
        self.dataset_dir = dataset_dir
        self.chunk_size_rows = chunk_size_rows

        with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.topology_id = int(metadata["topology_id"])
        self.dataset_id = int(metadata.get("dataset_id", self._infer_dataset_id(dataset_dir)))

    def _infer_dataset_id(self, dataset_dir: Path) -> int:
        name = dataset_dir.name
        if name.startswith("dataset_"):
            return int(name.split("_")[-1])
        raise ValueError(f"Could not infer dataset_id from directory name: {dataset_dir}")

    def __iter__(self) -> Iterator[StepTableBundle]:
        stream_iters: Dict[str, Iterator[StepChunk]] = {}
        current_chunks: Dict[str, Optional[StepChunk]] = {}

        for table_name, spec in self.TABLE_SPECS.items():
            file_path = self.dataset_dir / spec["relative_file"]
            if file_path.exists():
                stream = ParquetStepStream(
                    file_path=file_path,
                    columns=spec["columns"],
                    chunk_size_rows=self.chunk_size_rows,
                )
                stream_iters[table_name] = iter(stream)
                current_chunks[table_name] = self._advance_or_none(stream_iters[table_name])
            else:
                current_chunks[table_name] = None

        while True:
            active_steps = [
                chunk.step for chunk in current_chunks.values()
                if chunk is not None
            ]
            if not active_steps:
                break

            next_step = min(active_steps)
            tables_for_step: Dict[str, pl.DataFrame] = {}

            for table_name in self.TABLE_SPECS.keys():
                chunk = current_chunks[table_name]
                if chunk is not None and chunk.step == next_step:
                    tables_for_step[table_name] = chunk.df
                    if table_name in stream_iters:
                        current_chunks[table_name] = self._advance_or_none(stream_iters[table_name])
                    else:
                        current_chunks[table_name] = None
                else:
                    tables_for_step[table_name] = pl.DataFrame()

            yield StepTableBundle(
                dataset_id=self.dataset_id,
                topology_id=self.topology_id,
                step=next_step,
                tables=tables_for_step,
            )

    @staticmethod
    def _advance_or_none(iterator: Iterator[StepChunk]) -> Optional[StepChunk]:
        try:
            return next(iterator)
        except StopIteration:
            return None