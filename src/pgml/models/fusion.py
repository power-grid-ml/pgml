from __future__ import annotations

import torch
import torch.nn as nn


class NodeDeviceFusion(nn.Module):
    def __init__(self, node_static_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_static_encoder = (
            nn.Sequential(
                nn.Linear(node_static_dim, hidden_dim),
                nn.GELU(),
            )
            if node_static_dim > 0 else None
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
            self,
            node_static_x: torch.Tensor,
            node_measurement_latent: torch.Tensor,
            node_observability: torch.Tensor,
            device_latent: torch.Tensor,
            device_node_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = node_measurement_latent.shape[0]
        hidden_dim = node_measurement_latent.shape[1]

        if self.node_static_encoder is not None and node_static_x.shape[1] > 0:
            node_static_latent = self.node_static_encoder(node_static_x)
        else:
            node_static_latent = torch.zeros_like(node_measurement_latent)

        node_obs_latent = self.obs_encoder(node_observability)

        pooled_dtype = device_latent.dtype if device_latent.shape[0] > 0 else node_measurement_latent.dtype

        pooled_device = torch.zeros(
            (num_nodes, hidden_dim),
            dtype=pooled_dtype,
            device=node_measurement_latent.device,
        )

        if device_latent.shape[0] > 0:
            counts = torch.zeros((num_nodes, 1), dtype=pooled_dtype, device=device_latent.device)
            pooled_device.index_add_(0, device_node_index, device_latent)
            ones = torch.ones((device_latent.shape[0], 1), dtype=pooled_dtype, device=device_latent.device)
            counts.index_add_(0, device_node_index, ones)
            pooled_device = pooled_device / counts.clamp_min(1.0)

        x = torch.cat(
            [node_static_latent, node_measurement_latent, node_obs_latent, pooled_device],
            dim=-1
        )
        return self.fuse(x)