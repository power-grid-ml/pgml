from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import polars as pl
import pyarrow.parquet as pq


@dataclass
class StepChunk:
    step: int
    df: pl.DataFrame


class ParquetStepStream:
    """
    Streams one parquet table in row batches and yields complete step-grouped DataFrames.

    Assumptions:
    - parquet rows are ordered/grouped by `step`
    - rows for one step do not appear again after a later step has begun

    This allows a truly streaming step assembler without loading the full file.

    #TODO: Add a fallback mode for unsorted parquet files if needed in the future.
    """

    def __init__(
        self,
        file_path: Path,
        columns: Optional[list[str]] = None,
        chunk_size_rows: int = 50_000,
    ):
        self.file_path = file_path
        self.columns = columns
        self.chunk_size_rows = chunk_size_rows

    def __iter__(self) -> Iterator[StepChunk]:
        if not self.file_path.exists():
            return
            yield  # pragma: no cover

        parquet_file = pq.ParquetFile(self.file_path)
        carry_df: Optional[pl.DataFrame] = None

        for record_batch in parquet_file.iter_batches(
            batch_size=self.chunk_size_rows,
            columns=self.columns,
        ):
            chunk_df = pl.from_arrow(record_batch)
            if chunk_df.height == 0:
                continue

            if carry_df is not None and carry_df.height > 0:
                chunk_df = pl.concat([carry_df, chunk_df], how="vertical")
                carry_df = None

            if "step" not in chunk_df.columns:
                continue

            # We assume rows are grouped by step. We keep the final step in carry_df
            # because it may continue in the next batch.
            step_values = chunk_df["step"].to_list()
            if not step_values:
                continue

            last_step = int(step_values[-1])

            completed_df = chunk_df.filter(pl.col("step") != last_step)
            carry_df = chunk_df.filter(pl.col("step") == last_step)

            if completed_df.height > 0:
                grouped = completed_df.partition_by("step", as_dict=True)
                for step_value, step_df in grouped.items():
                    yield StepChunk(step=int(step_value[0]), df=step_df)

        if carry_df is not None and carry_df.height > 0:
            grouped = carry_df.partition_by("step", as_dict=True)
            for step_value, step_df in grouped.items():
                yield StepChunk(step=int(step_value[0]), df=step_df)