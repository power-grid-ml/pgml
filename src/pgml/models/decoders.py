from __future__ import annotations

import torch
import torch.nn as nn


class NodeDecoder(nn.Module):
    """
    Decodes updated node latents into node token reconstruction space.

    First implementation predicts one token payload per existing node token.

    Input:
    - node_latent:   [N, H]
    - target_tokens: [N, T, D]  only used for shaping by caller

    Output:
    - predicted_value: [N, T, D]

    #TODO: Replace repeated-token decoding with a true sequence/token decoder
    #      once transformer-based decoders are introduced.
    """

    def __init__(self, hidden_dim: int, out_value_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_value_dim),
        )

    def forward(self, node_latent: torch.Tensor, num_tokens: int) -> torch.Tensor:
        base = self.mlp(node_latent)  # [N, D]
        return base.unsqueeze(1).repeat(1, num_tokens, 1)


class EdgeDecoder(nn.Module):
    """
    Decodes edge latents into edge token reconstruction space.
    """

    def __init__(self, hidden_dim: int, out_value_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_value_dim),
        )

    def forward(self, edge_latent: torch.Tensor, num_tokens: int) -> torch.Tensor:
        base = self.mlp(edge_latent)
        return base.unsqueeze(1).repeat(1, num_tokens, 1)


class DeviceDecoder(nn.Module):
    """
    Decodes device latents into:
    - parameter token reconstruction
    - spectrum token reconstruction

    A simple device-type-specific head structure is used to avoid forcing the same
    output semantics across heterogeneous devices.

    #TODO: Replace token repetition decoding with a true token decoder.
    #TODO: Later add explicit injected-device heads once injected devices become
    #      first-class entities in topology and dataset assembly.
    """

    def __init__(
        self,
        hidden_dim: int,
        param_out_dim: int,
        spec_out_dim: int,
        num_device_types: int = 4,
    ):
        super().__init__()

        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, param_out_dim),
            )
            for _ in range(num_device_types)
        ])

        self.spec_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, spec_out_dim),
            )
            for _ in range(num_device_types)
        ])

    def forward(
        self,
        device_latent: torch.Tensor,
        device_type: torch.Tensor,
        num_param_tokens: int,
        num_spec_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device_latent.shape[0] == 0:
            param_pred = torch.zeros((0, num_param_tokens, self.param_heads[0][-1].out_features), device=device_latent.device)
            spec_pred = torch.zeros((0, num_spec_tokens, self.spec_heads[0][-1].out_features), device=device_latent.device)
            return param_pred, spec_pred

        param_base = []
        spec_base = []
        for i in range(device_latent.shape[0]):
            d_type = int(device_type[i].item())
            param_base.append(self.param_heads[d_type](device_latent[i:i + 1]))
            spec_base.append(self.spec_heads[d_type](device_latent[i:i + 1]))

        param_base = torch.cat(param_base, dim=0)
        spec_base = torch.cat(spec_base, dim=0)

        param_pred = param_base.unsqueeze(1).repeat(1, num_param_tokens, 1)
        spec_pred = spec_base.unsqueeze(1).repeat(1, num_spec_tokens, 1)

        return param_pred, spec_pred