from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import polars as pl
import torch
from torch_geometric.data import HeteroData

from pgml.data_pipeline.tokenizer import MeasurementTokenizer
from pgml.data_pipeline.topology import TopologyCache


class StepGraphAssembler:
    """
    Assembles one full HeteroData graph per (dataset_id, step).

    The graph contains:
    - static topology
    - node measurement tokens
    - edge measurement tokens
    - explicit device entities with parameter/spectrum tokens
    - parallel target tensors for supervised learning

    First implementation principle:
    Inputs and targets are identical clean tensors. Masking/noise injection will
    be introduced later as a dedicated training-time module.

    #TODO: Add dedicated noisy-input / clean-target separation once masking
    #      modules are introduced for node/edge/device inference training.
    """

    def __init__(
        self,
        topology_cache: TopologyCache,
        tokenizer: MeasurementTokenizer,
        node_feature_prefixes: tuple[str, ...] = ("v1", "v2", "v3"),
        edge_current_prefixes: tuple[str, ...] = ("i1", "i2", "i3"),
        edge_power_prefixes: tuple[str, ...] = (),
        spectrum_prefixes: tuple[str, ...] = ("spectrum1", "spectrum2", "spectrum3"),
    ):
        self.topology_cache = topology_cache
        self.tokenizer = tokenizer
        self.node_feature_prefixes = node_feature_prefixes
        self.edge_current_prefixes = edge_current_prefixes
        self.edge_power_prefixes = edge_power_prefixes
        self.spectrum_prefixes = spectrum_prefixes

    def assemble_step_graph(
        self,
        dataset_dir: Path,
        step: int,
    ) -> HeteroData:
        metadata = self._read_metadata(dataset_dir)
        topology_id = int(metadata["topology_id"])
        dataset_id = int(metadata.get("dataset_id", self._infer_dataset_id(dataset_dir)))

        base_graph = self.topology_cache.get_topology(topology_id)

        node_df = self._read_step_table(dataset_dir / "node_data" / "data.parquet", step)
        edge_df = self._read_step_table(dataset_dir / "edge_data" / "data.parquet", step)

        load_param_df = self._read_step_table(dataset_dir / "load_parameters" / "data.parquet", step)
        gen_param_df = self._read_step_table(dataset_dir / "generator_parameters" / "data.parquet", step)
        vsource_param_df = self._read_step_table(dataset_dir / "vsource_parameters" / "data.parquet", step)
        injected_param_df = self._read_step_table(dataset_dir / "injected_error_parameters" / "data.parquet", step)
        spectrum_df = self._read_step_table(dataset_dir / "spectrum" / "data.parquet", step)

        graph = base_graph.clone()

        # -------------------------
        # Node tokens
        # -------------------------
        node_ids = graph["node"].node_id.tolist()
        node_tokens = self.tokenizer.tokenize_node_measurements(
            df=node_df,
            entity_ids=node_ids,
            feature_prefixes=self.node_feature_prefixes,
        )

        graph["node"].meas_value = node_tokens.value
        graph["node"].meas_frequency = node_tokens.frequency
        graph["node"].meas_type = node_tokens.type_id
        graph["node"].meas_mask = node_tokens.mask

        # Current implementation uses same clean tensors as targets
        graph["target_node"].voltage_value = node_tokens.value.clone()
        graph["target_node"].voltage_frequency = node_tokens.frequency.clone()
        graph["target_node"].voltage_type = node_tokens.type_id.clone()
        graph["target_node"].voltage_mask = node_tokens.mask.clone()

        # -------------------------
        # Edge tokens
        # -------------------------
        edge_type = ("node", "physical", "node")
        edge_ids = graph[edge_type].edge_id.tolist()
        edge_tokens = self.tokenizer.tokenize_edge_measurements(
            df=edge_df,
            entity_ids=edge_ids,
            current_prefixes=self.edge_current_prefixes,
            power_prefixes=self.edge_power_prefixes,
        )

        graph["edge"].meas_value = edge_tokens.value
        graph["edge"].meas_frequency = edge_tokens.frequency
        graph["edge"].meas_type = edge_tokens.type_id
        graph["edge"].meas_mask = edge_tokens.mask

        graph["target_edge"].current_value = edge_tokens.value.clone()
        graph["target_edge"].current_frequency = edge_tokens.frequency.clone()
        graph["target_edge"].current_type = edge_tokens.type_id.clone()
        graph["target_edge"].current_mask = edge_tokens.mask.clone()

        # -------------------------
        # Device parameter tokens
        # -------------------------
        device_type = graph["device"].device_type
        device_id = graph["device"].device_id

        merged_param_df = self._merge_parameter_tables(
            load_param_df=load_param_df,
            gen_param_df=gen_param_df,
            vsource_param_df=vsource_param_df,
            injected_param_df=injected_param_df,
        )

        device_param_tokens = self.tokenizer.tokenize_device_parameters(
            param_df=merged_param_df,
            device_type=device_type,
            device_ids=device_id,
        )

        graph["device"].param_value = device_param_tokens.value
        graph["device"].param_frequency = device_param_tokens.frequency
        graph["device"].param_type = device_param_tokens.type_id
        graph["device"].param_mask = device_param_tokens.mask

        graph["target_device"].param_value = device_param_tokens.value.clone()
        graph["target_device"].param_frequency = device_param_tokens.frequency.clone()
        graph["target_device"].param_type = device_param_tokens.type_id.clone()
        graph["target_device"].param_mask = device_param_tokens.mask.clone()

        # -------------------------
        # Device spectrum tokens
        # -------------------------
        device_spectrum_tokens = self.tokenizer.tokenize_device_spectra(
            spectrum_df=spectrum_df,
            device_type=device_type,
            device_ids=device_id,
            spectrum_prefixes=self.spectrum_prefixes,
        )

        graph["device"].spec_value = device_spectrum_tokens.value
        graph["device"].spec_frequency = device_spectrum_tokens.frequency
        graph["device"].spec_type = device_spectrum_tokens.type_id
        graph["device"].spec_mask = device_spectrum_tokens.mask

        graph["target_device"].spec_value = device_spectrum_tokens.value.clone()
        graph["target_device"].spec_frequency = device_spectrum_tokens.frequency.clone()
        graph["target_device"].spec_type = device_spectrum_tokens.type_id.clone()
        graph["target_device"].spec_mask = device_spectrum_tokens.mask.clone()

        # -------------------------
        # Graph metadata
        # -------------------------
        graph.dataset_id = torch.tensor([dataset_id], dtype=torch.long)
        graph.topology_id = torch.tensor([topology_id], dtype=torch.long)
        graph.step = torch.tensor([step], dtype=torch.long)

        return graph

    def list_steps(self, dataset_dir: Path) -> list[int]:
        """
        Uses node_data as the canonical source of available simulation steps.

        #TODO: If future datasets contain missing node_data for some valid steps,
        #      build the step index across all available tables instead.
        """
        file_path = dataset_dir / "node_data" / "data.parquet"
        if not file_path.exists():
            return []

        df = pl.read_parquet(file_path, columns=["step"])
        return sorted(df["step"].unique().cast(pl.Int64).to_list())

    def _read_metadata(self, dataset_dir: Path) -> Dict:
        with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_dataset_id(self, dataset_dir: Path) -> int:
        name = dataset_dir.name
        if name.startswith("dataset_"):
            return int(name.split("_")[-1])
        raise ValueError(f"Could not infer dataset_id from directory name: {dataset_dir}")

    def _read_step_table(self, file_path: Path, step: int) -> pl.DataFrame:
        if not file_path.exists():
            return pl.DataFrame()

        df = pl.read_parquet(file_path)
        if "step" not in df.columns:
            return pl.DataFrame()

        return df.filter(pl.col("step") == step)

    def _merge_parameter_tables(
        self,
        load_param_df: pl.DataFrame,
        gen_param_df: pl.DataFrame,
        vsource_param_df: pl.DataFrame,
        injected_param_df: pl.DataFrame,
    ) -> pl.DataFrame:
        dfs = []
        for df in [load_param_df, gen_param_df, vsource_param_df, injected_param_df]:
            if df.height > 0:
                dfs.append(df)

        if not dfs:
            return pl.DataFrame()

        # Diagonal concat keeps all columns
        return pl.concat(dfs, how="diagonal")