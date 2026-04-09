import json
import logging
from pathlib import Path
from typing import List, Dict

import polars as pl

from pgml.config import config_dir, PipelineConfig, resource_dir

logging.basicConfig(level=logging.INFO)


class OutOfCoreStatsCompiler:
    """
    Streams Parquet data one file at a time (Map-Reduce pattern) to compute global
    scaling statistics without exceeding the memory footprint of a single chunk.
    """

    def __init__(self, data_dir: Path, train_dataset_ids: List[int], feature_specs: Dict[str, List[str]]):
        """
        Args:
            data_dir: Base directory containing dataset exports.
            train_dataset_ids: List of dataset IDs belonging to the training set.
            feature_specs: Dictionary mapping table names to the prefixes of polar columns.
                           Example: {"node_data":["v1", "v2", "v3"], "edge_data": ["i1", "i2", "i3"]}
        """
        self.data_dir = data_dir
        self.train_dataset_ids = train_dataset_ids
        self.feature_specs = feature_specs

    def _get_parquet_paths(self, table_name: str) -> List[Path]:
        paths = []
        for d_id in self.train_dataset_ids:
            p = self.data_dir / f"dataset_{d_id}" / table_name / "data.parquet"
            if p.exists():
                paths.append(p)
        return paths

    def _polar_to_rect_expr(self, prefix: str) -> List[pl.Expr]:
        mag_col = f"{prefix}"
        ang_col = f"{prefix}_angle"

        real_expr = (pl.col(mag_col) * pl.col(ang_col).cos()).alias(f"{prefix}_real")
        imag_expr = (pl.col(mag_col) * pl.col(ang_col).sin()).alias(f"{prefix}_imag")

        return [real_expr, imag_expr]

    def _compile_table_stats(self, table_name: str, prefixes: List[str]) -> Dict[str, Dict]:
        paths = self._get_parquet_paths(table_name)
        if not paths:
            logging.warning(f"No {table_name} found for training datasets.")
            return {}

        target_cols = []
        for p in prefixes:
            target_cols.extend([f"{p}_real", f"{p}_imag"])

        all_chunk_stats = []

        # --- MAP PHASE: Process one file at a time ---
        for path in paths:
            lf = pl.scan_parquet(path)

            # 1. Add rectangular columns
            rect_exprs = []
            for p in prefixes:
                rect_exprs.extend(self._polar_to_rect_expr(p))
            lf = lf.with_columns(rect_exprs)

            # 2. Define intermediate aggregation math
            agg_exprs = [pl.len().alias("count")]
            for col in target_cols:
                agg_exprs.extend([
                    pl.col(col).sum().alias(f"{col}_sum"),
                    (pl.col(col) ** 2).sum().alias(f"{col}_sum_sq"),
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).max().alias(f"{col}_max"),
                ])

            # 3. Execute lazy graph for this specific file ONLY
            chunk_df = lf.group_by("frequency").agg(agg_exprs).collect()
            all_chunk_stats.append(chunk_df)

        if not all_chunk_stats:
            return {}

        # --- REDUCE PHASE: Combine intermediate stats ---
        combined_lf = pl.concat(all_chunk_stats).lazy()

        reduce_exprs = [pl.col("count").sum().alias("total_count")]
        for col in target_cols:
            reduce_exprs.extend([
                pl.col(f"{col}_sum").sum().alias(f"{col}_sum_total"),
                pl.col(f"{col}_sum_sq").sum().alias(f"{col}_sum_sq_total"),
                pl.col(f"{col}_min").min().alias(f"{col}_min_global"),
                pl.col(f"{col}_max").max().alias(f"{col}_max_global"),
            ])

        global_stats = combined_lf.group_by("frequency").agg(reduce_exprs)

        # --- DERIVE FINAL MOMENTS (Mean, Std, Min, Max) ---
        final_exprs = [pl.col("frequency")]
        for col in target_cols:
            mean_expr = (pl.col(f"{col}_sum_total") / pl.col("total_count")).alias(f"{col}_mean")

            # Variance = (SumSq / N) - Mean^2
            # We clip to 0.0 because floating point precision can cause tiny negative variances
            var_expr = (
                    (pl.col(f"{col}_sum_sq_total") / pl.col("total_count")) - (mean_expr ** 2)
            ).clip(lower_bound=0.0)

            std_expr = var_expr.sqrt().alias(f"{col}_std")

            final_exprs.extend([
                mean_expr,
                std_expr,
                pl.col(f"{col}_min_global").alias(f"{col}_min"),
                pl.col(f"{col}_max_global").alias(f"{col}_max")
            ])

        final_df = global_stats.select(final_exprs).collect()

        # Convert to nested dictionary mapping
        stats_dict = {}
        for row in final_df.to_dicts():
            freq = str(int(row["frequency"]))
            stats_dict[freq] = {}
            for col in target_cols:
                stats_dict[freq][col] = {
                    "mean": row[f"{col}_mean"],
                    "std": row[f"{col}_std"] if row[f"{col}_std"] not in (None, 0.0) else 1.0,
                    "min": row[f"{col}_min"],
                    "max": row[f"{col}_max"]
                }

        return stats_dict

    def run_and_save(self, output_path: Path):
        logging.info("Computing out-of-core statistics via Map-Reduce...")

        full_stats = {}
        for table_name, prefixes in self.feature_specs.items():
            logging.info(f"Processing {table_name}...")
            full_stats[table_name] = self._compile_table_stats(table_name, prefixes)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_stats, f, indent=4)
        logging.info(f"Statistics saved to {output_path}")


if __name__ == "__main__":
    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)
    base_data = Path(resource_dir) / config.paths.input_dir

    # Define generically which tables to target and which polar magnitudes to parse
    specs = {
        "node_data": ["v1", "v2", "v3"],
        "edge_data": ["i1", "i2", "i3"],
        # Add generator_data, load_data, etc., here later if needed
    }

    compiler = OutOfCoreStatsCompiler(
        data_dir=base_data,
        train_dataset_ids=[2, 3],  # Make sure to include all training sets
        feature_specs=specs
    )

    compiler.run_and_save(base_data / "scaling_stats.json")