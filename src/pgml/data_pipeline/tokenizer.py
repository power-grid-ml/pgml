from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict

import polars as pl
import torch
import numpy as np


DEVICE_TYPE_MAP: Dict[str, int] = {
    "load": 0,
    "generator": 1,
    "vsource": 2,
    "injected": 3,
}

NODE_MEASUREMENT_TYPE_MAP: Dict[str, int] = {
    "voltage": 0,
}

EDGE_MEASUREMENT_TYPE_MAP: Dict[str, int] = {
    "current": 0,
    "power": 1,
}

DEVICE_TOKEN_TYPE_MAP: Dict[str, int] = {
    "param": 0,
    "spectrum": 1,
}


@dataclass(frozen=True)
class TokenizedBatch:
    """
    Generic padded token container for a set of entities.

    Shapes:
    - value:      [num_entities, max_tokens, value_dim]
    - frequency:  [num_entities, max_tokens]
    - type_id:    [num_entities, max_tokens]
    - mask:       [num_entities, max_tokens]
    """
    value: torch.Tensor
    frequency: torch.Tensor
    type_id: torch.Tensor
    mask: torch.Tensor


class MeasurementTokenizer:
    """
    Builds padded token tensors from per-row measurement tables using vectorized Polars operations.

    This implementation avoids row-by-row iteration and manual grouping, which is the primary
    bottleneck for training data collation.
    """

    def __init__(
            self,
            max_node_tokens: int = 60,
            max_edge_tokens: int = 60,
            max_device_param_tokens: int = 10,
            max_device_spec_tokens: int = 60
    ):
        self.max_node_tokens = max_node_tokens
        self.max_edge_tokens = max_edge_tokens
        self.max_device_param_tokens = max_device_param_tokens
        self.max_device_spec_tokens = max_device_spec_tokens

    def _get_preallocated_tensors(self, num_entities: int, max_tokens: int, value_dim: int):
        value = torch.zeros((num_entities, max_tokens, value_dim), dtype=torch.float32)
        frequency = torch.zeros((num_entities, max_tokens), dtype=torch.float32)
        type_id = torch.zeros((num_entities, max_tokens), dtype=torch.long)
        mask = torch.zeros((num_entities, max_tokens), dtype=torch.bool)
        return value, frequency, type_id, mask

    def tokenize_node_measurements(
        self,
        df: pl.DataFrame,
        entity_ids: Sequence[int],
        feature_prefixes: Sequence[str],
    ) -> TokenizedBatch:
        num_entities = len(entity_ids)
        max_tokens = self.max_node_tokens
        value_dim = 2 * len(feature_prefixes)
        
        value, frequency, type_id, mask = self._get_preallocated_tensors(num_entities, max_tokens, value_dim)
        
        if df.height == 0 or num_entities == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        id_list = [int(i) for i in entity_ids]
        id_to_idx = {id_val: i for i, id_val in enumerate(id_list)}
        
        df_proc = (
            df.filter(pl.col("node_id").is_in(id_list))
            .sort("frequency")
            .with_columns([
                pl.col("node_id").replace(id_to_idx, default=None).cast(pl.Int64).alias("entity_idx"),
                pl.int_range(0, pl.len(), dtype=pl.Int64).over("node_id").alias("rank")
            ])
            .filter(pl.col("rank") < max_tokens)
        )
        
        if df_proc.height == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        e_idx = df_proc["entity_idx"].to_numpy()
        r_idx = df_proc["rank"].to_numpy()
        
        frequency[e_idx, r_idx] = torch.from_numpy(df_proc["frequency"].to_numpy().astype("float32"))
        
        for i, prefix in enumerate(feature_prefixes):
            mags = torch.from_numpy(df_proc[prefix].fill_null(0.0).to_numpy().astype("float32"))
            angs = torch.from_numpy(df_proc[f"{prefix}_angle"].fill_null(0.0).to_numpy().astype("float32"))
            value[e_idx, r_idx, 2*i] = mags * torch.cos(angs)
            value[e_idx, r_idx, 2*i + 1] = mags * torch.sin(angs)
            
        type_id[e_idx, r_idx] = NODE_MEASUREMENT_TYPE_MAP["voltage"]
        mask[e_idx, r_idx] = True
        
        return TokenizedBatch(value, frequency, type_id, mask)

    def tokenize_edge_measurements(
        self,
        df: pl.DataFrame,
        entity_ids: Sequence[int],
        current_prefixes: Sequence[str],
        power_prefixes: Optional[Sequence[str]] = None,
    ) -> TokenizedBatch:
        num_entities = len(entity_ids)
        max_tokens = self.max_edge_tokens
        power_prefixes = power_prefixes or []
        value_dim = 2 * (len(current_prefixes) + len(power_prefixes))
        
        value, frequency, type_id, mask = self._get_preallocated_tensors(num_entities, max_tokens, value_dim)
        
        if df.height == 0 or num_entities == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        id_list = [int(i) for i in entity_ids]
        id_to_idx = {id_val: i for i, id_val in enumerate(id_list)}
        
        df_proc = (
            df.filter(pl.col("edge_id").is_in(id_list))
            .sort("frequency")
            .with_columns([
                pl.col("edge_id").replace(id_to_idx, default=None).cast(pl.Int64).alias("entity_idx"),
                pl.int_range(0, pl.len(), dtype=pl.Int64).over("edge_id").alias("rank")
            ])
            .filter(pl.col("rank") < max_tokens)
        )
        
        if df_proc.height == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        e_idx = df_proc["entity_idx"].to_numpy()
        r_idx = df_proc["rank"].to_numpy()
        
        frequency[e_idx, r_idx] = torch.from_numpy(df_proc["frequency"].to_numpy().astype("float32"))
        
        # Determine token type: if power prefixes are present, we label as power (1)
        # following original logic.
        t_id = EDGE_MEASUREMENT_TYPE_MAP["current"]
        if power_prefixes:
             t_id = EDGE_MEASUREMENT_TYPE_MAP["power"]

        curr_offset = 0
        for i, prefix in enumerate(current_prefixes):
            mags = torch.from_numpy(df_proc[prefix].fill_null(0.0).to_numpy().astype("float32"))
            angs = torch.from_numpy(df_proc[f"{prefix}_angle"].fill_null(0.0).to_numpy().astype("float32"))
            value[e_idx, r_idx, 2*i] = mags * torch.cos(angs)
            value[e_idx, r_idx, 2*i + 1] = mags * torch.sin(angs)
            curr_offset = 2 * (i + 1)
            
        for i, prefix in enumerate(power_prefixes):
            mags = torch.from_numpy(df_proc[prefix].fill_null(0.0).to_numpy().astype("float32"))
            angs = torch.from_numpy(df_proc[f"{prefix}_angle"].fill_null(0.0).to_numpy().astype("float32"))
            value[e_idx, r_idx, curr_offset + 2*i] = mags * torch.cos(angs)
            value[e_idx, r_idx, curr_offset + 2*i + 1] = mags * torch.sin(angs)

        type_id[e_idx, r_idx] = t_id
        mask[e_idx, r_idx] = True
        
        return TokenizedBatch(value, frequency, type_id, mask)

    def tokenize_device_parameters(
        self,
        param_df: pl.DataFrame,
        device_type: torch.Tensor,
        device_ids: torch.Tensor,
    ) -> TokenizedBatch:
        num_devices = device_ids.shape[0]
        max_tokens = self.max_device_param_tokens
        
        value, frequency, type_id, mask = self._get_preallocated_tensors(num_devices, max_tokens, 1)
        
        if param_df.height == 0 or num_devices == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        # Map device type/id to entity index
        device_ident = pl.DataFrame({
            "type_idx": device_type.numpy().astype(np.int64),
            "id_idx": device_ids.numpy().astype(np.int64),
            "entity_idx": np.arange(num_devices, dtype=np.int64)
        })

        load_cols = ["p1", "q1", "p2", "q2", "p3", "q3"]
        gen_cols = ["p1", "q1", "p2", "q2", "p3", "q3"]
        vsource_cols = ["pu1", "pu2", "pu3"]
        injected_cols = ["sc1_mva"]
        
        melted_parts = []
        
        specs = [
            ("load_id", DEVICE_TYPE_MAP["load"], load_cols),
            ("generator_id", DEVICE_TYPE_MAP["generator"], gen_cols),
            ("vsource_id", DEVICE_TYPE_MAP["vsource"], vsource_cols),
            ("node_id", DEVICE_TYPE_MAP["injected"], injected_cols),
        ]
        
        for id_col, t_idx, cols in specs:
            if id_col in param_df.columns:
                df_part = param_df.filter(pl.col(id_col).is_not_null())
                if df_part.height > 0:
                    avail_cols = [c for c in cols if c in df_part.columns]
                    m = df_part.select([id_col, *avail_cols]).melt(id_vars=id_col, value_name="val")
                    m = m.filter(pl.col("val").is_not_null()).select([
                        pl.lit(t_idx).alias("type_idx"),
                        pl.col(id_col).alias("id_idx"),
                        pl.col("val")
                    ])
                    melted_parts.append(m)

        if not melted_parts:
            return TokenizedBatch(value, frequency, type_id, mask)
            
        full_melted = pl.concat(melted_parts)
        res = full_melted.join(device_ident, on=["type_idx", "id_idx"], how="inner")
        
        if res.height == 0:
            return TokenizedBatch(value, frequency, type_id, mask)
            
        res = res.with_columns(
            pl.int_range(0, pl.len(), dtype=pl.Int64).over("entity_idx").alias("rank")
        ).filter(pl.col("rank") < max_tokens)
        
        e_idx = res["entity_idx"].to_numpy()
        r_idx = res["rank"].to_numpy()
        
        value[e_idx, r_idx, 0] = torch.from_numpy(res["val"].to_numpy().astype("float32"))
        type_id[e_idx, r_idx] = DEVICE_TOKEN_TYPE_MAP["param"]
        mask[e_idx, r_idx] = True
        
        return TokenizedBatch(value, frequency, type_id, mask)

    def tokenize_device_spectra(
        self,
        spectrum_df: pl.DataFrame,
        device_type: torch.Tensor,
        device_ids: torch.Tensor,
        spectrum_prefixes: Sequence[str] = ("spectrum1", "spectrum2", "spectrum3"),
    ) -> TokenizedBatch:
        num_devices = device_ids.shape[0]
        max_tokens = self.max_device_spec_tokens
        value_dim = 2 * len(spectrum_prefixes)
        
        value, frequency, type_id, mask = self._get_preallocated_tensors(num_devices, max_tokens, value_dim)
        
        if spectrum_df.height == 0 or num_devices == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        parent_type_map = {
            "load": DEVICE_TYPE_MAP["load"],
            "generator": DEVICE_TYPE_MAP["generator"],
            "vsource": DEVICE_TYPE_MAP["vsource"],
        }

        device_ident = pl.DataFrame({
            "type_idx": device_type.numpy().astype(np.int64),
            "id_idx": device_ids.numpy().astype(np.int64),
            "entity_idx": np.arange(num_devices, dtype=np.int64)
        })

        df_proc = (
            spectrum_df.with_columns(
                pl.col("parent_type").replace(parent_type_map, default=None).cast(pl.Int64).alias("type_idx")
            )
            .filter(pl.col("type_idx").is_not_null())
            .rename({"parent_id": "id_idx"})
            .join(device_ident, on=["type_idx", "id_idx"], how="inner")
            .sort("frequency")
            .with_columns(
                pl.int_range(0, pl.len(), dtype=pl.Int64).over("entity_idx").alias("rank")
            )
            .filter(pl.col("rank") < max_tokens)
        )

        if df_proc.height == 0:
            return TokenizedBatch(value, frequency, type_id, mask)

        e_idx = df_proc["entity_idx"].to_numpy()
        r_idx = df_proc["rank"].to_numpy()
        
        frequency[e_idx, r_idx] = torch.from_numpy(df_proc["frequency"].to_numpy().astype("float32"))
        
        for i, prefix in enumerate(spectrum_prefixes):
            mags = torch.from_numpy(df_proc[prefix].fill_null(0.0).to_numpy().astype("float32"))
            angs = torch.from_numpy(df_proc[f"{prefix}_angle"].fill_null(0.0).to_numpy().astype("float32"))
            value[e_idx, r_idx, 2*i] = mags * torch.cos(angs)
            value[e_idx, r_idx, 2*i + 1] = mags * torch.sin(angs)

        type_id[e_idx, r_idx] = DEVICE_TOKEN_TYPE_MAP["spectrum"]
        mask[e_idx, r_idx] = True

        return TokenizedBatch(value, frequency, type_id, mask)
