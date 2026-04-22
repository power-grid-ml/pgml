from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import polars as pl


DEFAULT_TABLES = [
    "node_data/data.parquet",
    "edge_data/data.parquet",
    "load_parameters/data.parquet",
    "generator_parameters/data.parquet",
    "vsource_parameters/data.parquet",
    "injected_error_parameters/data.parquet",
    "spectrum/data.parquet",
]


def _read_step_column(file_path: Path) -> Optional[pl.Series]:
    if not file_path.exists():
        return None

    schema_names = pl.scan_parquet(file_path).collect_schema().names()
    if "step" not in schema_names:
        return None

    return pl.scan_parquet(file_path).select("step").collect(streaming=True)["step"]


def is_sorted_by_step(file_path: Path) -> bool:
    """
    Returns True if the parquet file is sorted non-decreasingly by `step`.
    Files without a `step` column are treated as True.
    """
    step_series = _read_step_column(file_path)
    if step_series is None or len(step_series) <= 1:
        return True

    step_values = step_series.cast(pl.Int64).to_list()
    return all(step_values[i] <= step_values[i + 1] for i in range(len(step_values) - 1))


def sort_parquet_by_step(
    file_path: Path,
    secondary_sort_columns: Optional[list[str]] = None,
    overwrite: bool = True,
) -> bool:
    """
    Sorts one parquet file by `step` and optional secondary columns.

    Returns:
        True  -> file was rewritten
        False -> file already sorted or has no step column / does not exist
    """
    if not file_path.exists():
        return False

    lf = pl.scan_parquet(file_path)
    schema_names = lf.collect_schema().names()
    if "step" not in schema_names:
        return False

    if is_sorted_by_step(file_path):
        return False

    sort_cols = ["step"]
    if secondary_sort_columns:
        sort_cols.extend([c for c in secondary_sort_columns if c in schema_names])

    df = lf.collect(streaming=True).sort(sort_cols)

    target_path = file_path if overwrite else file_path.with_name(f"{file_path.stem}_sorted{file_path.suffix}")
    df.write_parquet(target_path)

    return True


def validate_and_sort_dataset_dir(
    dataset_dir: Path,
    table_paths: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> dict[str, str]:
    """
    Checks all relevant parquet files inside one dataset directory.
    If a file is not sorted by step, rewrites it sorted by step.

    Returns a status dictionary:
        "missing"
        "no_step_column"
        "already_sorted"
        "sorted_now"
    """
    table_paths = list(table_paths or DEFAULT_TABLES)
    statuses: dict[str, str] = {}

    secondary_sort_map = {
        "node_data/data.parquet": ["node_id", "frequency"],
        "edge_data/data.parquet": ["edge_id", "frequency"],
        "load_parameters/data.parquet": ["load_id"],
        "generator_parameters/data.parquet": ["generator_id"],
        "vsource_parameters/data.parquet": ["vsource_id"],
        "injected_error_parameters/data.parquet": ["node_id"],
        "spectrum/data.parquet": ["parent_type", "parent_id", "frequency"],
    }

    for rel_path in table_paths:
        file_path = dataset_dir / rel_path

        if not file_path.exists():
            statuses[rel_path] = "missing"
            if verbose:
                print(f"[SORT CHECK] {file_path}: missing")
            continue

        schema_names = pl.scan_parquet(file_path).collect_schema().names()
        if "step" not in schema_names:
            statuses[rel_path] = "no_step_column"
            if verbose:
                print(f"[SORT CHECK] {file_path}: no step column")
            continue

        if is_sorted_by_step(file_path):
            statuses[rel_path] = "already_sorted"
            if verbose:
                print(f"[SORT CHECK] {file_path}: already sorted")
        else:
            sort_parquet_by_step(
                file_path=file_path,
                secondary_sort_columns=secondary_sort_map.get(rel_path, None),
                overwrite=True,
            )
            statuses[rel_path] = "sorted_now"
            if verbose:
                print(f"[SORT CHECK] {file_path}: sorted now")

    return statuses


def validate_and_sort_all_datasets(
    base_data_dir: Path,
    dataset_ids: Optional[Iterable[int]] = None,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """
    Validates/sorts all datasets.

    If dataset_ids is None, auto-discovers directories named dataset_*.
    """
    if dataset_ids is None:
        dataset_dirs = sorted(
            [p for p in base_data_dir.iterdir() if p.is_dir() and p.name.startswith("dataset_")]
        )
    else:
        dataset_dirs = [base_data_dir / f"dataset_{dataset_id}" for dataset_id in dataset_ids]

    all_statuses: dict[str, dict[str, str]] = {}
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            if verbose:
                print(f"[SORT CHECK] {dataset_dir}: missing dataset directory")
            continue

        if verbose:
            print(f"\n=== Checking dataset: {dataset_dir.name} ===")

        all_statuses[dataset_dir.name] = validate_and_sort_dataset_dir(
            dataset_dir=dataset_dir,
            verbose=verbose,
        )

    return all_statuses


if __name__ == "__main__":
    from pgml.config import config_dir, PipelineConfig, resource_dir
    from pgml.training.main import _resolve_rel_abs_path
    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)
    input_dir = _resolve_rel_abs_path(resource_dir, config.paths.input_dir)
    validate_and_sort_all_datasets(input_dir, verbose=True)