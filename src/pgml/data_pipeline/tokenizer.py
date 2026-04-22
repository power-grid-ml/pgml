from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict

import polars as pl
import torch


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
    Builds padded token tensors from per-row measurement tables.

    This first implementation keeps the token payload deliberately simple:
    - electrical complex values are represented in rectangular form [real, imag]
    - scalar parameters are represented as [value]
    - frequencies are passed separately
    - token type ids are passed separately

    #TODO: Extend token payload to include optional magnitude/angle, source-quality
    #      flags, or learned metadata embeddings if later needed.
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

    def polar_to_rect(self, magnitudes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        real_parts = magnitudes * torch.cos(angles)
        imag_parts = magnitudes * torch.sin(angles)
        rect_tensor = torch.empty((magnitudes.shape[0], magnitudes.shape[1] * 2), dtype=torch.float32)
        rect_tensor[:, 0::2] = real_parts
        rect_tensor[:, 1::2] = imag_parts
        return rect_tensor

    def pad_token_sequences(
            self,
            token_lists: List[torch.Tensor],
            freq_lists: List[torch.Tensor],
            type_lists: List[torch.Tensor],
            value_dim: int,
            max_tokens_override: int,
    ) -> TokenizedBatch:
        num_entities = len(token_lists)
        # FIX: Pad to global max_tokens so all graphs match shapes for PyG batching
        max_tokens = max_tokens_override

        value = torch.zeros((num_entities, max_tokens, value_dim), dtype=torch.float32)
        frequency = torch.zeros((num_entities, max_tokens), dtype=torch.float32)
        type_id = torch.zeros((num_entities, max_tokens), dtype=torch.long)
        mask = torch.zeros((num_entities, max_tokens), dtype=torch.bool)

        for i, (tok, freq, typ) in enumerate(zip(token_lists, freq_lists, type_lists)):
            n = min(tok.shape[0], max_tokens)  # truncate if exceeds limit
            if n == 0:
                continue
            value[i, :n] = tok[:n]
            frequency[i, :n] = freq[:n]
            type_id[i, :n] = typ[:n]
            mask[i, :n] = True

        return TokenizedBatch(value=value, frequency=frequency, type_id=type_id, mask=mask)

    def tokenize_node_measurements(
        self,
        df: pl.DataFrame,
        entity_ids: Sequence[int],
        feature_prefixes: Sequence[str],
    ) -> TokenizedBatch:
        """
        Tokenizes node voltage harmonics from node_data.

        Expected columns:
        - node_id
        - frequency
        - v1, v1_angle, v2, v2_angle, v3, v3_angle, ...
        """
        grouped = {int(node_id): [] for node_id in entity_ids}

        if df.height > 0:
            for row in df.iter_rows(named=True):
                node_id = int(row["node_id"])
                if node_id not in grouped:
                    continue

                freq = float(row["frequency"])
                mags = []
                angs = []
                for prefix in feature_prefixes:
                    mags.append(float(row.get(prefix, 0.0) or 0.0))
                    angs.append(float(row.get(f"{prefix}_angle", 0.0) or 0.0))

                mags_t = torch.tensor(mags, dtype=torch.float32).unsqueeze(0)
                angs_t = torch.tensor(angs, dtype=torch.float32).unsqueeze(0)
                rect = self.polar_to_rect(mags_t, angs_t).squeeze(0)  # [2 * num_phases]

                grouped[node_id].append((rect, freq))

        token_lists: List[torch.Tensor] = []
        freq_lists: List[torch.Tensor] = []
        type_lists: List[torch.Tensor] = []

        type_id_value = NODE_MEASUREMENT_TYPE_MAP["voltage"]
        value_dim = 2 * len(feature_prefixes)

        for node_id in entity_ids:
            tokens = grouped[int(node_id)]
            if tokens:
                tok_tensor = torch.stack([t[0] for t in tokens], dim=0)
                freq_tensor = torch.tensor([t[1] for t in tokens], dtype=torch.float32)
                type_tensor = torch.full((len(tokens),), type_id_value, dtype=torch.long)
            else:
                tok_tensor = torch.zeros((0, value_dim), dtype=torch.float32)
                freq_tensor = torch.zeros((0,), dtype=torch.float32)
                type_tensor = torch.zeros((0,), dtype=torch.long)

            token_lists.append(tok_tensor)
            freq_lists.append(freq_tensor)
            type_lists.append(type_tensor)

        return self.pad_token_sequences(token_lists, freq_lists, type_lists, value_dim, self.max_node_tokens)

    def tokenize_edge_measurements(
        self,
        df: pl.DataFrame,
        entity_ids: Sequence[int],
        current_prefixes: Sequence[str],
        power_prefixes: Optional[Sequence[str]] = None,
    ) -> TokenizedBatch:
        """
        Tokenizes edge measurements from edge_data.

        First implementation stores each frequency row as a single token. If both
        current and power are present, they are concatenated into one token.

        #TODO: Consider splitting current and power into separate tokens if later
        #      experiments show that token-type disentanglement helps learning.
        """
        power_prefixes = power_prefixes or []
        grouped = {int(edge_id): [] for edge_id in entity_ids}

        if df.height > 0:
            for row in df.iter_rows(named=True):
                edge_id = int(row["edge_id"])
                if edge_id not in grouped:
                    continue

                freq = float(row["frequency"])

                mags = []
                angs = []

                token_type = EDGE_MEASUREMENT_TYPE_MAP["current"]

                for prefix in current_prefixes:
                    mags.append(float(row.get(prefix, 0.0) or 0.0))
                    angs.append(float(row.get(f"{prefix}_angle", 0.0) or 0.0))

                for prefix in power_prefixes:
                    mags.append(float(row.get(prefix, 0.0) or 0.0))
                    angs.append(float(row.get(f"{prefix}_angle", 0.0) or 0.0))
                    token_type = EDGE_MEASUREMENT_TYPE_MAP["power"]

                mags_t = torch.tensor(mags, dtype=torch.float32).unsqueeze(0)
                angs_t = torch.tensor(angs, dtype=torch.float32).unsqueeze(0)
                rect = self.polar_to_rect(mags_t, angs_t).squeeze(0)

                grouped[edge_id].append((rect, freq, token_type))

        token_lists: List[torch.Tensor] = []
        freq_lists: List[torch.Tensor] = []
        type_lists: List[torch.Tensor] = []

        value_dim = 2 * (len(current_prefixes) + len(power_prefixes))

        for edge_id in entity_ids:
            tokens = grouped[int(edge_id)]
            if tokens:
                tok_tensor = torch.stack([t[0] for t in tokens], dim=0)
                freq_tensor = torch.tensor([t[1] for t in tokens], dtype=torch.float32)
                type_tensor = torch.tensor([t[2] for t in tokens], dtype=torch.long)
            else:
                tok_tensor = torch.zeros((0, value_dim), dtype=torch.float32)
                freq_tensor = torch.zeros((0,), dtype=torch.float32)
                type_tensor = torch.zeros((0,), dtype=torch.long)

            token_lists.append(tok_tensor)
            freq_lists.append(freq_tensor)
            type_lists.append(type_tensor)

        return self.pad_token_sequences(token_lists, freq_lists, type_lists, value_dim, self.max_edge_tokens)

    def tokenize_device_parameters(
        self,
        param_df: pl.DataFrame,
        device_type: torch.Tensor,
        device_ids: torch.Tensor,
    ) -> TokenizedBatch:
        """
        Tokenizes per-device scalar parameter tables.

        Each scalar parameter becomes one token with value_dim = 1.

        Matching keys:
        - load         -> load_id
        - generator    -> generator_id
        - vsource      -> vsource_id
        - injected     -> node_id  (for injected_error_parameters)

        #TODO: Later replace scalar-only tokens with richer tokens carrying
        #      parameter-name embeddings explicitly in the value payload.
        """
        num_devices = int(device_ids.shape[0])
        grouped: List[List[tuple[float, float, int]]] = [[] for _ in range(num_devices)]

        if param_df.height == 0 or num_devices == 0:
            return self.pad_token_sequences(
                token_lists=[torch.zeros((0, 1), dtype=torch.float32) for _ in range(num_devices)],
                freq_lists=[torch.zeros((0,), dtype=torch.float32) for _ in range(num_devices)],
                type_lists=[torch.zeros((0,), dtype=torch.long) for _ in range(num_devices)],
                value_dim=1,
                max_tokens_override=self.max_device_param_tokens
            )

        load_cols = ["p1", "q1", "p2", "q2", "p3", "q3"]
        gen_cols = ["p1", "q1", "p2", "q2", "p3", "q3"]
        vsource_cols = ["pu1", "pu2", "pu3"]
        injected_cols = ["sc1_mva"]

        # Build fast lookup by (device_type, domain_id)
        device_lookup = {
            (int(device_type[i].item()), int(device_ids[i].item())): i
            for i in range(num_devices)
        }

        for row in param_df.iter_rows(named=True):
            matched = False

            if "load_id" in row and row["load_id"] is not None:
                key = (DEVICE_TYPE_MAP["load"], int(row["load_id"]))
                if key in device_lookup:
                    idx = device_lookup[key]
                    for col_name in load_cols:
                        if col_name in row and row[col_name] is not None:
                            grouped[idx].append((float(row[col_name]), 0.0, DEVICE_TOKEN_TYPE_MAP["param"]))
                    matched = True

            if "generator_id" in row and row["generator_id"] is not None:
                key = (DEVICE_TYPE_MAP["generator"], int(row["generator_id"]))
                if key in device_lookup:
                    idx = device_lookup[key]
                    for col_name in gen_cols:
                        if col_name in row and row[col_name] is not None:
                            grouped[idx].append((float(row[col_name]), 0.0, DEVICE_TOKEN_TYPE_MAP["param"]))
                    matched = True

            if "vsource_id" in row and row["vsource_id"] is not None:
                key = (DEVICE_TYPE_MAP["vsource"], int(row["vsource_id"]))
                if key in device_lookup:
                    idx = device_lookup[key]
                    for col_name in vsource_cols:
                        if col_name in row and row[col_name] is not None:
                            grouped[idx].append((float(row[col_name]), 0.0, DEVICE_TOKEN_TYPE_MAP["param"]))
                    matched = True

            if "node_id" in row and row["node_id"] is not None and not matched:
                key = (DEVICE_TYPE_MAP["injected"], int(row["node_id"]))
                if key in device_lookup:
                    idx = device_lookup[key]
                    for col_name in injected_cols:
                        if col_name in row and row[col_name] is not None:
                            grouped[idx].append((float(row[col_name]), 0.0, DEVICE_TOKEN_TYPE_MAP["param"]))

        token_lists: List[torch.Tensor] = []
        freq_lists: List[torch.Tensor] = []
        type_lists: List[torch.Tensor] = []

        for items in grouped:
            if items:
                tok_tensor = torch.tensor([[x[0]] for x in items], dtype=torch.float32)
                freq_tensor = torch.tensor([x[1] for x in items], dtype=torch.float32)
                type_tensor = torch.tensor([x[2] for x in items], dtype=torch.long)
            else:
                tok_tensor = torch.zeros((0, 1), dtype=torch.float32)
                freq_tensor = torch.zeros((0,), dtype=torch.float32)
                type_tensor = torch.zeros((0,), dtype=torch.long)

            token_lists.append(tok_tensor)
            freq_lists.append(freq_tensor)
            type_lists.append(type_tensor)

        return self.pad_token_sequences(token_lists, freq_lists, type_lists, 1, self.max_device_param_tokens)

    def tokenize_device_spectra(
        self,
        spectrum_df: pl.DataFrame,
        device_type: torch.Tensor,
        device_ids: torch.Tensor,
        spectrum_prefixes: Sequence[str] = ("spectrum1", "spectrum2", "spectrum3"),
    ) -> TokenizedBatch:
        """
        Tokenizes injected spectra. Each spectrum row becomes one token carrying
        rectangular complex values for all available phases/components.

        Parent matching:
        - parent_type in {'load', 'generator', 'vsource'}
        - direct injections are intentionally deferred until an explicit injected
          device representation is added to topology.

        #TODO: Add explicit synthetic "injected" devices attached to nodes so that
        #      direct injected_error_parameters and injected/node spectra can be
        #      represented uniformly as device entities.
        """
        num_devices = int(device_ids.shape[0])
        grouped: List[List[tuple[torch.Tensor, float, int]]] = [[] for _ in range(num_devices)]

        if spectrum_df.height == 0 or num_devices == 0:
            value_dim = 2 * len(spectrum_prefixes)
            return self.pad_token_sequences(
                token_lists=[torch.zeros((0, value_dim), dtype=torch.float32) for _ in range(num_devices)],
                freq_lists=[torch.zeros((0,), dtype=torch.float32) for _ in range(num_devices)],
                type_lists=[torch.zeros((0,), dtype=torch.long) for _ in range(num_devices)],
                value_dim=value_dim,
                max_tokens_override=self.max_device_spec_tokens
            )

        parent_type_map = {
            "load": DEVICE_TYPE_MAP["load"],
            "generator": DEVICE_TYPE_MAP["generator"],
            "vsource": DEVICE_TYPE_MAP["vsource"],
        }

        device_lookup = {
            (int(device_type[i].item()), int(device_ids[i].item())): i
            for i in range(num_devices)
        }

        for row in spectrum_df.iter_rows(named=True):
            parent_type = row["parent_type"]
            if parent_type not in parent_type_map:
                continue

            key = (parent_type_map[parent_type], int(row["parent_id"]))
            if key not in device_lookup:
                continue

            idx = device_lookup[key]
            freq = float(row["frequency"])

            mags = []
            angs = []
            for prefix in spectrum_prefixes:
                mags.append(float(row.get(prefix, 0.0) or 0.0))
                angs.append(float(row.get(f"{prefix}_angle", 0.0) or 0.0))

            mags_t = torch.tensor(mags, dtype=torch.float32).unsqueeze(0)
            angs_t = torch.tensor(angs, dtype=torch.float32).unsqueeze(0)
            rect = self.polar_to_rect(mags_t, angs_t).squeeze(0)

            grouped[idx].append((rect, freq, DEVICE_TOKEN_TYPE_MAP["spectrum"]))

        token_lists: List[torch.Tensor] = []
        freq_lists: List[torch.Tensor] = []
        type_lists: List[torch.Tensor] = []

        value_dim = 2 * len(spectrum_prefixes)

        for items in grouped:
            if items:
                tok_tensor = torch.stack([x[0] for x in items], dim=0)
                freq_tensor = torch.tensor([x[1] for x in items], dtype=torch.float32)
                type_tensor = torch.tensor([x[2] for x in items], dtype=torch.long)
            else:
                tok_tensor = torch.zeros((0, value_dim), dtype=torch.float32)
                freq_tensor = torch.zeros((0,), dtype=torch.float32)
                type_tensor = torch.zeros((0,), dtype=torch.long)

            token_lists.append(tok_tensor)
            freq_lists.append(freq_tensor)
            type_lists.append(type_tensor)

        return self.pad_token_sequences(token_lists, freq_lists, type_lists, value_dim, self.max_device_spec_tokens)