from pathlib import Path
from typing import Dict

import polars as pl
import torch
from torch_geometric.data import HeteroData


class TopologyCache:
    """
    Loads and caches static node and edge features for various grid topologies.
    Constructs the base HeteroData objects that dynamic features will be appended to.
    """

    def __init__(self, base_data_dir: Path):
        self.base_dir = base_data_dir
        self.cache: Dict[int, HeteroData] = {}

    def get_topology(self, topology_id: int) -> HeteroData:
        if topology_id in self.cache:
            return self.cache[topology_id].clone()

        topo_dir = self.base_dir / f"topology_{topology_id}"

        # Load Static Nodes
        node_df = pl.read_parquet(topo_dir / "static_node_features.parquet")
        # Ensure consistent sorting by internal node_id
        node_df = node_df.sort("node_id")
        node_features = torch.tensor(
            node_df.drop(["id", "node_id"]).to_numpy(), dtype=torch.float32
        )

        # Load Static Edges
        edge_df = pl.read_parquet(topo_dir / "static_edge_features.parquet")

        # PyG expects edge_index of shape [2, num_edges]
        edge_index = torch.tensor(
            edge_df.select(["node_id1", "node_id2"]).to_numpy().T, dtype=torch.int64
        )
        edge_features = torch.tensor(
            edge_df.drop(["id", "edge_id", "node_id1", "node_id2"]).to_numpy(), dtype=torch.float32
        )

        # Build Base HeteroData
        data = HeteroData()
        data['node'].static_x = node_features

        edge_type = ('node', 'physical', 'node')
        data[edge_type].edge_index = edge_index
        data[edge_type].static_edge_attr = edge_features

        self.cache[topology_id] = data
        return data.clone()