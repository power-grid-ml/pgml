from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import torch
from torch_geometric.data import HeteroData


class TopologyCache:
    """
    Loads and caches static node, edge, and explicit device features for various
    grid topologies. Constructs the base HeteroData objects that dynamic features
    will later be appended to.

    Returned structure:
    - data["node"].static_x
    - data["node"].node_id
    - data[("node", "physical", "node")].edge_index
    - data[("node", "physical", "node")].static_edge_attr
    - data[("node", "physical", "node")].edge_id
    - data["device"].static_x
    - data["device"].node_index
    - data["device"].device_type
    - data["device"].device_id
    """

    def __init__(self, base_data_dir: Path):
        self.base_dir = Path(base_data_dir)
        self.cache: Dict[int, HeteroData] = {}

    def get_topology(self, topology_id: int) -> HeteroData:
        if topology_id in self.cache:
            return self.cache[topology_id].clone()

        topo_dir = self.base_dir / f"topology_{topology_id}"

        node_df = self._load_nodes(topo_dir)
        edge_df = self._load_edges(topo_dir)
        device_df = self._load_and_merge_devices(topo_dir)

        data = HeteroData()

        # -------------------------
        # Nodes
        # -------------------------
        node_df = node_df.sort("node_id")

        node_ids: List[int] = node_df["node_id"].cast(pl.Int64).to_list()
        node_id_to_index = {int(node_id): idx for idx, node_id in enumerate(node_ids)}

        node_feature_df = node_df.drop(["id", "node_id"], strict=False)
        node_features = torch.tensor(node_feature_df.to_numpy(), dtype=torch.float32)

        data["node"].static_x = node_features
        data["node"].node_id = torch.tensor(node_ids, dtype=torch.long)

        # -------------------------
        # Physical edges
        # -------------------------
        edge_df = edge_df.sort("edge_id")

        edge_pairs = edge_df.select(["node_id1", "node_id2"]).rows()
        mapped_edge_index = torch.tensor(
            [
                [node_id_to_index[int(src)], node_id_to_index[int(dst)]]
                for src, dst in edge_pairs
            ],
            dtype=torch.long,
        ).T.contiguous()

        edge_feature_df = edge_df.drop(["id", "edge_id", "node_id1", "node_id2"], strict=False)
        edge_features = torch.tensor(edge_feature_df.to_numpy(), dtype=torch.float32)

        edge_type = ("node", "physical", "node")
        data[edge_type].edge_index = mapped_edge_index
        data[edge_type].static_edge_attr = edge_features
        data[edge_type].edge_id = torch.tensor(
            edge_df["edge_id"].cast(pl.Int64).to_list(),
            dtype=torch.long
        )

        # -------------------------
        # Explicit devices
        # -------------------------
        if device_df is not None and device_df.height > 0:
            device_df = device_df.sort(["node_id", "device_type", "device_id"])

            device_node_ids = device_df["node_id"].cast(pl.Int64).to_list()
            device_node_index = torch.tensor(
                [node_id_to_index[int(node_id)] for node_id in device_node_ids],
                dtype=torch.long
            )

            device_type = torch.tensor(
                device_df["device_type"].cast(pl.Int64).to_list(),
                dtype=torch.long
            )
            device_id = torch.tensor(
                device_df["device_id"].cast(pl.Int64).to_list(),
                dtype=torch.long
            )

            device_feature_exclude = ["id", "node_id", "device_type", "device_id"]
            device_feature_cols = [c for c in device_df.columns if c not in device_feature_exclude]

            if device_feature_cols:
                device_static_x = torch.tensor(
                    device_df.select(device_feature_cols).to_numpy(),
                    dtype=torch.float32
                )
            else:
                device_static_x = torch.zeros((device_df.height, 0), dtype=torch.float32)

            data["device"].static_x = device_static_x
            data["device"].node_index = device_node_index
            data["device"].device_type = device_type
            data["device"].device_id = device_id
        else:
            data["device"].static_x = torch.zeros((0, 0), dtype=torch.float32)
            data["device"].node_index = torch.zeros((0,), dtype=torch.long)
            data["device"].device_type = torch.zeros((0,), dtype=torch.long)
            data["device"].device_id = torch.zeros((0,), dtype=torch.long)

        self.cache[topology_id] = data
        return data.clone()

    def _load_nodes(self, topo_dir: Path) -> pl.DataFrame:
        node_file = topo_dir / "static_node_features.parquet"
        if not node_file.exists():
            raise FileNotFoundError(f"Missing static node file: {node_file}")
        return pl.read_parquet(node_file)

    def _load_edges(self, topo_dir: Path) -> pl.DataFrame:
        edge_file = topo_dir / "static_edge_features.parquet"
        if not edge_file.exists():
            raise FileNotFoundError(f"Missing static edge file: {edge_file}")
        return pl.read_parquet(edge_file)

    def _load_and_merge_devices(self, topo_dir: Path) -> Optional[pl.DataFrame]:
        """
        Loads explicit device parquet files and merges them into a single device dataframe
        with a unified schema.

        Output columns include:
        - id
        - node_id
        - device_type
        - device_id
        - numeric feature columns
        """
        frames: List[pl.DataFrame] = []

        file_specs = [
            ("static_load_features.parquet", "load_id", 0),
            ("static_generator_features.parquet", "generator_id", 1),
            ("static_vsource_features.parquet", "vsource_id", 2),
        ]

        for filename, source_id_col, device_type_value in file_specs:
            file_path = topo_dir / filename
            if not file_path.exists():
                continue

            df = pl.read_parquet(file_path)
            if df.height == 0:
                continue

            df = self._normalize_device_df(
                df=df,
                source_id_col=source_id_col,
                device_type_value=device_type_value,
            )
            frames.append(df)

        if not frames:
            return None

        all_cols = sorted(set().union(*(set(df.columns) for df in frames)))
        normalized_frames: List[pl.DataFrame] = []

        for df in frames:
            missing_cols = [c for c in all_cols if c not in df.columns]
            if missing_cols:
                fill_exprs = []
                for col_name in missing_cols:
                    if col_name in {"id", "node_id", "device_type", "device_id"}:
                        fill_exprs.append(pl.lit(0).cast(pl.Int64).alias(col_name))
                    else:
                        fill_exprs.append(pl.lit(0.0).cast(pl.Float32).alias(col_name))
                df = df.with_columns(fill_exprs)

            df = df.select(all_cols)
            normalized_frames.append(df)

        merged = pl.concat(normalized_frames, how="vertical")

        cast_exprs = []
        for col_name in merged.columns:
            if col_name in {"id", "node_id", "device_type", "device_id"}:
                cast_exprs.append(pl.col(col_name).cast(pl.Int64))
            else:
                cast_exprs.append(pl.col(col_name).cast(pl.Float32).fill_null(0.0))

        return merged.with_columns(cast_exprs)

    def _normalize_device_df(
        self,
        df: pl.DataFrame,
        source_id_col: str,
        device_type_value: int,
    ) -> pl.DataFrame:
        """
        Renames the source-specific identifier into unified `device_id`
        and ensures a valid `device_type` column exists.
        """
        if source_id_col not in df.columns:
            raise ValueError(f"Expected column '{source_id_col}' not found in device dataframe.")

        if "device_type" in df.columns:
            pass
        elif "device_type_idx" in df.columns:
            df = df.rename({"device_type_idx": "device_type"})
        else:
            df = df.with_columns(
                pl.lit(device_type_value).cast(pl.Int64).alias("device_type")
            )

        df = df.rename({source_id_col: "device_id"})

        # TODO: Preserve original source-specific ids in extra columns if inverse mapping
        #       becomes important for debugging or reporting.
        return df