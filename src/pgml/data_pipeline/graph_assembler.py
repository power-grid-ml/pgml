from __future__ import annotations

import polars as pl
import torch
from torch_geometric.data import HeteroData

from pgml.data_pipeline.tokenizer import MeasurementTokenizer
from pgml.data_pipeline.multi_table_step_stream import StepTableBundle
from pgml.data_pipeline.topology import TopologyCache


class GraphAssembler:
    """
    Assembles one full HeteroData graph from a streamed StepTableBundle.

    This version is compatible with true chunked streaming and no longer performs
    any parquet I/O itself.
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

    def assemble_graph(self, bundle: StepTableBundle) -> HeteroData:
        base_graph = self.topology_cache.get_topology(bundle.topology_id)
        graph = base_graph.clone()

        node_df = bundle.tables["node_data"]
        edge_df = bundle.tables["edge_data"]
        load_param_df = bundle.tables["load_parameters"]
        gen_param_df = bundle.tables["generator_parameters"]
        vsource_param_df = bundle.tables["vsource_parameters"]
        injected_param_df = bundle.tables["injected_error_parameters"]
        spectrum_df = bundle.tables["spectrum"]

        # -------------------------
        # Node tokens
        # -------------------------
        node_ids = graph["node"].node_id.tolist()
        node_tokens = self.tokenizer.tokenize_node_measurements(
            df=node_df, entity_ids=node_ids, feature_prefixes=self.node_feature_prefixes,
        )

        graph["node"].meas_value = node_tokens.value
        graph["node"].meas_frequency = node_tokens.frequency
        graph["node"].meas_type = node_tokens.type_id
        graph["node"].meas_mask = node_tokens.mask

        graph["node"].target_voltage_value = node_tokens.value.clone()
        graph["node"].target_voltage_frequency = node_tokens.frequency.clone()
        graph["node"].target_voltage_type = node_tokens.type_id.clone()
        graph["node"].target_voltage_mask = node_tokens.mask.clone()

        # -------------------------
        # Edge tokens
        # -------------------------
        edge_type = ("node", "physical", "node")
        edge_ids = graph[edge_type].edge_id.tolist()
        edge_tokens = self.tokenizer.tokenize_edge_measurements(
            df=edge_df, entity_ids=edge_ids, current_prefixes=self.edge_current_prefixes,
            power_prefixes=self.edge_power_prefixes,
        )

        graph[edge_type].meas_value = edge_tokens.value
        graph[edge_type].meas_frequency = edge_tokens.frequency
        graph[edge_type].meas_type = edge_tokens.type_id
        graph[edge_type].meas_mask = edge_tokens.mask

        graph[edge_type].target_current_value = edge_tokens.value.clone()
        graph[edge_type].target_current_frequency = edge_tokens.frequency.clone()
        graph[edge_type].target_current_type = edge_tokens.type_id.clone()
        graph[edge_type].target_current_mask = edge_tokens.mask.clone()

        # -------------------------
        # Device parameter tokens
        # -------------------------
        device_type = graph["device"].device_type
        device_id = graph["device"].device_id

        merged_param_df = self._merge_parameter_tables(load_param_df, gen_param_df, vsource_param_df, injected_param_df)

        device_param_tokens = self.tokenizer.tokenize_device_parameters(
            param_df=merged_param_df, device_type=device_type, device_ids=device_id,
        )

        graph["device"].param_value = device_param_tokens.value
        graph["device"].param_frequency = device_param_tokens.frequency
        graph["device"].param_type = device_param_tokens.type_id
        graph["device"].param_mask = device_param_tokens.mask

        graph["device"].target_param_value = device_param_tokens.value.clone()
        graph["device"].target_param_frequency = device_param_tokens.frequency.clone()
        graph["device"].target_param_type = device_param_tokens.type_id.clone()
        graph["device"].target_param_mask = device_param_tokens.mask.clone()

        # -------------------------
        # Device spectrum tokens
        # -------------------------
        device_spectrum_tokens = self.tokenizer.tokenize_device_spectra(
            spectrum_df=spectrum_df, device_type=device_type, device_ids=device_id,
            spectrum_prefixes=self.spectrum_prefixes,
        )

        graph["device"].spec_value = device_spectrum_tokens.value
        graph["device"].spec_frequency = device_spectrum_tokens.frequency
        graph["device"].spec_type = device_spectrum_tokens.type_id
        graph["device"].spec_mask = device_spectrum_tokens.mask

        graph["device"].target_spec_value = device_spectrum_tokens.value.clone()
        graph["device"].target_spec_frequency = device_spectrum_tokens.frequency.clone()
        graph["device"].target_spec_type = device_spectrum_tokens.type_id.clone()
        graph["device"].target_spec_mask = device_spectrum_tokens.mask.clone()

        # -------------------------
        # Graph metadata
        # -------------------------
        graph.dataset_id = torch.tensor([bundle.dataset_id], dtype=torch.long)
        graph.topology_id = torch.tensor([bundle.topology_id], dtype=torch.long)
        graph.step = torch.tensor([bundle.step], dtype=torch.long)

        return graph

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

        return pl.concat(dfs, how="diagonal")