from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMeanPooling(nn.Module):
    """
    Mean pooling over the token dimension with boolean mask support.

    Inputs:
    - x:    [N, T, D]
    - mask: [N, T]
    Output:
    - y:    [N, D]
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype, device=x.device)

        mask_f = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom


class TokenValueEncoder(nn.Module):
    """
    Encodes token values together with frequency and token type embeddings.

    Input:
    - value:     [N, T, value_dim]
    - frequency: [N, T]
    - type_id:   [N, T]

    Output:
    - token embeddings [N, T, hidden_dim]

    #TODO: Replace the simple frequency MLP with richer harmonic/frequency
    #      embeddings if non-uniform frequency grids become important.
    """

    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        num_token_types: int,
        type_emb_dim: int = 8,
        freq_emb_dim: int = 8,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_token_types, type_emb_dim)
        self.freq_mlp = nn.Sequential(
            nn.Linear(1, freq_emb_dim),
            nn.GELU(),
            nn.Linear(freq_emb_dim, freq_emb_dim),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(value_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + type_emb_dim + freq_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        value: torch.Tensor,
        frequency: torch.Tensor,
        type_id: torch.Tensor,
    ) -> torch.Tensor:
        value_emb = self.value_mlp(value)
        type_emb = self.type_embedding(type_id)
        freq_emb = self.freq_mlp(frequency.unsqueeze(-1))

        x = torch.cat([value_emb, type_emb, freq_emb], dim=-1)
        return self.out(x)


class NodeMeasurementEncoder(nn.Module):
    """
    Encodes padded node measurement tokens into one latent vector per node.
    """

    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        num_token_types: int = 8,
    ):
        super().__init__()
        self.token_encoder = TokenValueEncoder(
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            num_token_types=num_token_types,
        )
        self.pool = MaskedMeanPooling()

    def forward(
        self,
        value: torch.Tensor,
        frequency: torch.Tensor,
        type_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self.token_encoder(value, frequency, type_id)
        return self.pool(token_emb, mask)


class EdgeMeasurementEncoder(nn.Module):
    """
    Encodes padded edge measurement tokens into one latent vector per edge.
    """

    def __init__(
        self,
        value_dim: int,
        hidden_dim: int,
        num_token_types: int = 8,
    ):
        super().__init__()
        self.token_encoder = TokenValueEncoder(
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            num_token_types=num_token_types,
        )
        self.pool = MaskedMeanPooling()

    def forward(
        self,
        value: torch.Tensor,
        frequency: torch.Tensor,
        type_id: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self.token_encoder(value, frequency, type_id)
        return self.pool(token_emb, mask)


class DeviceEncoder(nn.Module):
    """
    Encodes each explicit device into one latent vector using:
    - static device features
    - parameter token set
    - spectrum token set
    - device type embedding

    Output:
    - device latent [num_devices, hidden_dim]
    """

    def __init__(
        self,
        static_dim: int,
        param_value_dim: int,
        spec_value_dim: int,
        hidden_dim: int,
        num_device_types: int = 8,
        num_token_types: int = 8,
        device_type_emb_dim: int = 8,
    ):
        super().__init__()

        self.device_type_embedding = nn.Embedding(num_device_types, device_type_emb_dim)

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim) if static_dim > 0 else nn.Identity(),
            nn.GELU() if static_dim > 0 else nn.Identity(),
        )

        self.param_token_encoder = TokenValueEncoder(
            value_dim=param_value_dim,
            hidden_dim=hidden_dim,
            num_token_types=num_token_types,
        )
        self.param_pool = MaskedMeanPooling()

        self.spec_token_encoder = TokenValueEncoder(
            value_dim=spec_value_dim,
            hidden_dim=hidden_dim,
            num_token_types=num_token_types,
        )
        self.spec_pool = MaskedMeanPooling()

        in_dim = hidden_dim + hidden_dim + hidden_dim + device_type_emb_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        static_x: torch.Tensor,
        device_type: torch.Tensor,
        param_value: torch.Tensor,
        param_frequency: torch.Tensor,
        param_type: torch.Tensor,
        param_mask: torch.Tensor,
        spec_value: torch.Tensor,
        spec_frequency: torch.Tensor,
        spec_type: torch.Tensor,
        spec_mask: torch.Tensor,
    ) -> torch.Tensor:
        if static_x.shape[0] == 0:
            return torch.zeros((0, self.fuse[-1].out_features), dtype=torch.float32, device=device_type.device)

        if static_x.shape[1] > 0:
            static_latent = self.static_encoder(static_x)
        else:
            hidden_dim = self.fuse[-1].out_features
            static_latent = torch.zeros((static_x.shape[0], hidden_dim), dtype=torch.float32, device=static_x.device)

        param_emb = self.param_token_encoder(param_value, param_frequency, param_type)
        param_latent = self.param_pool(param_emb, param_mask)

        spec_emb = self.spec_token_encoder(spec_value, spec_frequency, spec_type)
        spec_latent = self.spec_pool(spec_emb, spec_mask)

        type_latent = self.device_type_embedding(device_type)

        x = torch.cat([static_latent, param_latent, spec_latent, type_latent], dim=-1)
        return self.fuse(x)