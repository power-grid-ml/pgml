import json
import logging
from pathlib import Path
from typing import List, Dict

import polars as pl

logging.basicConfig(level=logging.INFO)


class OutOfCoreStatsCompiler:
    """
    Streams Parquet data using Polars LazyFrames to compute scaling statistics
    (Mean, Std) without loading the entire dataset into memory.
    """

    def __init__(self, data_dir: Path, train_dataset_ids: List[int]):
        self.data_dir = data_dir
        self.train_dataset_ids = train_dataset_ids

    def _get_parquet_paths(self, table_name: str) -> List[Path]:
        paths = []
        for d_id in self.train_dataset_ids:
            p = self.data_dir / f"dataset_{d_id}" / table_name / "data.parquet"
            if p.exists():
                paths.append(p)
        return paths

    def _polar_to_rect_expr(self, prefix: str) -> List[pl.Expr]:
        """
        Generates Polars expressions to convert Polar (mag, angle) to Rectangular (real, imag).
        Assumes angle is in radians.
        """
        mag_col = f"{prefix}"
        ang_col = f"{prefix}_angle"

        real_expr = (pl.col(mag_col) * pl.col(ang_col).cos()).alias(f"{prefix}_real")
        imag_expr = (pl.col(mag_col) * pl.col(ang_col).sin()).alias(f"{prefix}_imag")

        return [real_expr, imag_expr]

    def compute_node_statistics(self) -> Dict[str, Dict]:
        """
        Computes mean and std for node voltages, grouped by frequency.
        """
        paths = self._get_parquet_paths("node_data")
        if not paths:
            logging.warning("No node_data found for training datasets.")
            return {}

        # Scan all parquet files into a single LazyFrame
        lf = pl.scan_parquet(paths)

        # Convert V1, V2, V3 to rectangular
        rect_exprs = []
        for phase in ["v1", "v2", "v3"]:
            rect_exprs.extend(self._polar_to_rect_expr(phase))

        lf = lf.with_columns(rect_exprs)

        # Define columns to aggregate
        target_cols = [
            "v1_real", "v1_imag",
            "v2_real", "v2_imag",
            "v3_real", "v3_imag"
        ]

        # Group by frequency and compute mean/std
        agg_exprs = []
        for col in target_cols:
            agg_exprs.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std")
            ])

        stats_df = lf.group_by("frequency").agg(agg_exprs).collect()

        # Convert to nested dictionary: {frequency: {col_name: {'mean': val, 'std': val}}}
        stats_dict = {}
        for row in stats_df.to_dicts():
            freq = str(float(row["frequency"]))  # JSON keys must be strings
            stats_dict[freq] = {}
            for col in target_cols:
                stats_dict[freq][col] = {
                    "mean": row[f"{col}_mean"],
                    "std": row[f"{col}_std"] if row[f"{col}_std"] is not None else 1.0
                }

        return stats_dict

    def run_and_save(self, output_path: Path):
        logging.info("Computing out-of-core statistics for Node Data...")
        node_stats = self.compute_node_statistics()

        # Extensible to edge_data, load_data, etc.
        full_stats = {
            "node_data": node_stats
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_stats, f, indent=4)
        logging.info(f"Statistics saved to {output_path}")


if __name__ == "__main__":
    # Example execution
    base_data = Path("./data/export")
    compiler = OutOfCoreStatsCompiler(data_dir=base_data, train_dataset_ids=[2, 3, 4])
    compiler.run_and_save(base_data / "scaling_stats.json")