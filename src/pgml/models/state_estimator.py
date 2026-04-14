from __future__ import annotations

import torch
import torch.nn as nn

from pgml.models.decoders import DeviceDecoder, EdgeDecoder, NodeDecoder
from pgml.models.fusion import NodeDeviceFusion
from pgml.models.graph_state_estimator import GraphStateEstimator
from pgml.models.masking import DeviceInputNoiser, LatentObservabilityMasker
from pgml.models.token_encoders import DeviceEncoder, EdgeMeasurementEncoder, NodeMeasurementEncoder


class MultiModalStateEstimator(nn.Module):
    """
    End-to-end model for the hierarchical explicit-device architecture.

    Pipeline:
    1. encode node measurement token sets
    2. encode edge measurement token sets
    3. corrupt device inputs during training if requested
    4. encode explicit devices
    5. apply latent observability masking to node/edge measurements
    6. fuse device context into nodes
    7. optionally run graph state estimator
    8. decode node / edge / device targets
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
        self.hidden_dim = hidden_dim

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

        self.node_masker = LatentObservabilityMasker(hidden_dim=hidden_dim)
        self.edge_masker = LatentObservabilityMasker(hidden_dim=hidden_dim)
        self.device_noiser = DeviceInputNoiser(
            param_value_dim=device_param_value_dim,
            spec_value_dim=device_spec_value_dim,
        )

        self.node_device_fusion = NodeDeviceFusion(
            node_static_dim=node_static_dim,
            hidden_dim=hidden_dim,
        )

        self.edge_static_encoder = (
            nn.Sequential(
                nn.Linear(edge_static_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if edge_static_dim > 0 else None
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
            num_token_types=16,
        )

    def forward(
        self,
        batch,
        node_mask_ratio: float = 0.0,
        edge_mask_ratio: float = 0.0,
        device_noise_scale: float = 0.0,
        spectrum_drop_prob: float = 0.0,
        bypass_gnn: bool = False,
    ) -> dict[str, torch.Tensor]:
        edge_type = ("node", "physical", "node")

        noisy_param_value, noisy_spec_value = self.device_noiser(
            param_value=batch["device"].param_value,
            param_mask=batch["device"].param_mask,
            spec_value=batch["device"].spec_value,
            spec_mask=batch["device"].spec_mask,
            noise_scale=device_noise_scale,
            spectrum_drop_prob=spectrum_drop_prob,
        )

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
            param_value=noisy_param_value,
            param_frequency=batch["device"].param_frequency,
            param_type=batch["device"].param_type,
            param_mask=batch["device"].param_mask,
            spec_value=noisy_spec_value,
            spec_frequency=batch["device"].spec_frequency,
            spec_type=batch["device"].spec_type,
            spec_mask=batch["device"].spec_mask,
        )

        masked_node_latent, node_obs_indicator = self.node_masker(
            latent=node_meas_latent,
            mask_ratio=node_mask_ratio,
        )
        masked_edge_latent, edge_obs_indicator = self.edge_masker(
            latent=edge_meas_latent,
            mask_ratio=edge_mask_ratio,
        )

        node_latent = self.node_device_fusion(
            node_static_x=batch["node"].static_x,
            node_measurement_latent=masked_node_latent,
            node_observability=node_obs_indicator,
            device_latent=device_latent,
            device_node_index=batch["device"].node_index,
        )

        if self.edge_static_encoder is not None and batch[edge_type].static_edge_attr.shape[1] > 0:
            edge_static_latent = self.edge_static_encoder(batch[edge_type].static_edge_attr)
        else:
            edge_static_latent = torch.zeros_like(masked_edge_latent)

        edge_latent = masked_edge_latent + edge_static_latent

        if bypass_gnn:
            updated_node_latent = node_latent
        else:
            updated_node_latent = self.graph_estimator(
                node_latent=node_latent,
                edge_index=batch[edge_type].edge_index,
                edge_latent=edge_latent,
            )

        if device_latent.shape[0] > 0:
            conditioned_device_latent = device_latent + updated_node_latent[batch["device"].node_index]
        else:
            conditioned_device_latent = device_latent

        pred_node_value = self.node_decoder(
            node_latent=updated_node_latent,
            target_frequency=batch["target_node"].voltage_frequency,
            target_type=batch["target_node"].voltage_type,
        )

        pred_edge_value = self.edge_decoder(
            edge_latent=edge_latent,
            target_frequency=batch["target_edge"].current_frequency,
            target_type=batch["target_edge"].current_type,
        )

        pred_device_param, pred_device_spec = self.device_decoder(
            device_latent=conditioned_device_latent,
            device_type=batch["device"].device_type,
            param_frequency=batch["target_device"].param_frequency,
            param_type=batch["target_device"].param_type,
            spec_frequency=batch["target_device"].spec_frequency,
            spec_type=batch["target_device"].spec_type,
        )

        return {
            "pred_node_value": pred_node_value,
            "pred_edge_value": pred_edge_value,
            "pred_device_param": pred_device_param,
            "pred_device_spec": pred_device_spec,
            "node_latent": updated_node_latent,
            "edge_latent": edge_latent,
            "device_latent": conditioned_device_latent,
            "node_observability": node_obs_indicator,
            "edge_observability": edge_obs_indicator,
            "bypass_gnn": torch.tensor([1 if bypass_gnn else 0], device=updated_node_latent.device),
        }