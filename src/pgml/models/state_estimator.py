from __future__ import annotations

import torch
import torch.nn as nn

from pgml.models.decoders import DeviceDecoder, EdgeDecoder, NodeDecoder
from pgml.models.fusion import NodeDeviceFusion
from pgml.models.graph_state_estimator import GraphStateEstimator
from pgml.models.token_encoders import DeviceEncoder, EdgeMeasurementEncoder, NodeMeasurementEncoder


class MultiModalStateEstimator(nn.Module):
    """
    First end-to-end model for the new architecture.

    Pipeline:
    1. encode node measurement token sets
    2. encode edge measurement token sets
    3. encode explicit devices
    4. fuse device context into nodes
    5. run graph state estimator
    6. decode node / edge / device targets

    Outputs are returned as a dictionary for flexible multitask losses.
    """

    def __init__(
        self,
        node_static_dim: int,
        edge_static_dim: int,
        device_static_dim: int,
        node_value_dim: int,
        edge_value_dim: int,
        device_param_value_dim: int,
        device_spec_value_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.node_encoder = NodeMeasurementEncoder(
            value_dim=node_value_dim,
            hidden_dim=hidden_dim,
            num_token_types=16,
        )
        self.edge_encoder = EdgeMeasurementEncoder(
            value_dim=edge_value_dim,
            hidden_dim=hidden_dim,
            num_token_types=16,
        )
        self.device_encoder = DeviceEncoder(
            static_dim=device_static_dim,
            param_value_dim=device_param_value_dim,
            spec_value_dim=device_spec_value_dim,
            hidden_dim=hidden_dim,
            num_device_types=8,
            num_token_types=16,
        )

        self.node_device_fusion = NodeDeviceFusion(
            node_static_dim=node_static_dim,
            hidden_dim=hidden_dim,
        )

        self.edge_static_encoder = nn.Sequential(
            nn.Linear(edge_static_dim, hidden_dim) if edge_static_dim > 0 else nn.Identity(),
            nn.GELU() if edge_static_dim > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim) if edge_static_dim > 0 else nn.Identity(),
        )

        self.graph_estimator = GraphStateEstimator(
            hidden_dim=hidden_dim,
            num_layers=2,
            heads=4,
            dropout=0.1,
        )

        self.node_decoder = NodeDecoder(hidden_dim=hidden_dim, out_value_dim=node_value_dim)
        self.edge_decoder = EdgeDecoder(hidden_dim=hidden_dim, out_value_dim=edge_value_dim)
        self.device_decoder = DeviceDecoder(
            hidden_dim=hidden_dim,
            param_out_dim=device_param_value_dim,
            spec_out_dim=device_spec_value_dim,
            num_device_types=4,
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        edge_type = ("node", "physical", "node")

        # -------------------------
        # Encode local entities
        # -------------------------
        node_meas_latent = self.node_encoder(
            value=batch["node"].meas_value,
            frequency=batch["node"].meas_frequency,
            type_id=batch["node"].meas_type,
            mask=batch["node"].meas_mask,
        )

        edge_meas_latent = self.edge_encoder(
            value=batch["edge"].meas_value,
            frequency=batch["edge"].meas_frequency,
            type_id=batch["edge"].meas_type,
            mask=batch["edge"].meas_mask,
        )

        device_latent = self.device_encoder(
            static_x=batch["device"].static_x,
            device_type=batch["device"].device_type,
            param_value=batch["device"].param_value,
            param_frequency=batch["device"].param_frequency,
            param_type=batch["device"].param_type,
            param_mask=batch["device"].param_mask,
            spec_value=batch["device"].spec_value,
            spec_frequency=batch["device"].spec_frequency,
            spec_type=batch["device"].spec_type,
            spec_mask=batch["device"].spec_mask,
        )

        # -------------------------
        # Fuse node context
        # -------------------------
        node_latent = self.node_device_fusion(
            node_static_x=batch["node"].static_x,
            node_measurement_latent=node_meas_latent,
            device_latent=device_latent,
            device_node_index=batch["device"].node_index,
        )

        if batch[edge_type].static_edge_attr.shape[1] > 0:
            edge_static_latent = self.edge_static_encoder(batch[edge_type].static_edge_attr)
        else:
            edge_static_latent = torch.zeros_like(edge_meas_latent)

        edge_latent = edge_meas_latent + edge_static_latent

        # -------------------------
        # Graph propagation
        # -------------------------
        updated_node_latent = self.graph_estimator(
            node_latent=node_latent,
            edge_index=batch[edge_type].edge_index,
            edge_latent=edge_latent,
        )

        # -------------------------
        # Decode targets
        # -------------------------
        node_num_tokens = batch["target_node"].voltage_value.shape[1]
        edge_num_tokens = batch["target_edge"].current_value.shape[1]
        device_param_num_tokens = batch["target_device"].param_value.shape[1]
        device_spec_num_tokens = batch["target_device"].spec_value.shape[1]

        pred_node_value = self.node_decoder(updated_node_latent, num_tokens=node_num_tokens)
        pred_edge_value = self.edge_decoder(edge_latent, num_tokens=edge_num_tokens)

        # Device decoder uses graph-conditioned node context
        if device_latent.shape[0] > 0:
            conditioned_device_latent = device_latent + updated_node_latent[batch["device"].node_index]
        else:
            conditioned_device_latent = device_latent

        pred_device_param, pred_device_spec = self.device_decoder(
            device_latent=conditioned_device_latent,
            device_type=batch["device"].device_type,
            num_param_tokens=device_param_num_tokens,
            num_spec_tokens=device_spec_num_tokens,
        )

        return {
            "pred_node_value": pred_node_value,
            "pred_edge_value": pred_edge_value,
            "pred_device_param": pred_device_param,
            "pred_device_spec": pred_device_spec,
            "node_latent": updated_node_latent,
            "edge_latent": edge_latent,
            "device_latent": conditioned_device_latent,
        }