from __future__ import annotations

import torch
import torch.nn as nn

from pgml.models.token_encoders import TokenValueEncoder


class TokenConditionedDecoder(nn.Module):
    """
    Decodes entity latents into token-wise outputs conditioned on target token metadata.

    Inputs:
    - entity_latent: [N, H]
    - target_frequency: [N, T]
    - target_type: [N, T]

    Output:
    - predicted_value: [N, T, out_value_dim]

    Strategy:
    - embed target token metadata
    - concatenate repeated entity latent with target-token embedding
    - predict token-specific output

    This is the first replacement for the simplistic repeated-token decoder.

    #TODO: Upgrade to cross-attention decoding if later experiments show that
    #      repeated conditioning is insufficient for dense harmonic structure.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_value_dim: int,
        num_token_types: int = 16,
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
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + type_emb_dim + freq_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_value_dim),
        )

    def forward(
        self,
        entity_latent: torch.Tensor,
        target_frequency: torch.Tensor,
        target_type: torch.Tensor,
    ) -> torch.Tensor:
        n, t = target_frequency.shape
        if n == 0:
            out_dim = self.mlp[-1].out_features
            return torch.zeros((0, t, out_dim), dtype=torch.float32, device=target_frequency.device)

        freq_emb = self.freq_mlp(target_frequency.unsqueeze(-1))
        type_emb = self.type_embedding(target_type)

        repeated_latent = entity_latent.unsqueeze(1).expand(-1, t, -1)
        x = torch.cat([repeated_latent, freq_emb, type_emb], dim=-1)
        return self.mlp(x)


class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_value_dim: int, num_token_types: int = 16):
        super().__init__()
        self.decoder = TokenConditionedDecoder(
            hidden_dim=hidden_dim,
            out_value_dim=out_value_dim,
            num_token_types=num_token_types,
        )

    def forward(
        self,
        node_latent: torch.Tensor,
        target_frequency: torch.Tensor,
        target_type: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(node_latent, target_frequency, target_type)


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_value_dim: int, num_token_types: int = 16):
        super().__init__()
        self.decoder = TokenConditionedDecoder(
            hidden_dim=hidden_dim,
            out_value_dim=out_value_dim,
            num_token_types=num_token_types,
        )

    def forward(
        self,
        edge_latent: torch.Tensor,
        target_frequency: torch.Tensor,
        target_type: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(edge_latent, target_frequency, target_type)


class DeviceDecoder(nn.Module):
    """
    Device-type-specific token-conditioned decoder.

    A separate head per device type avoids forcing identical output semantics for:
    - load
    - generator
    - vsource
    - injected (reserved)

    TODO: Later add explicit injected-device heads once injected devices become
          first-class entities in topology and dataset assembly.
    """

    def __init__(
        self,
        hidden_dim: int,
        param_out_dim: int,
        spec_out_dim: int,
        num_device_types: int = 4,
        num_token_types: int = 16,
    ):
        super().__init__()

        self.param_heads = nn.ModuleList([
            TokenConditionedDecoder(
                hidden_dim=hidden_dim,
                out_value_dim=param_out_dim,
                num_token_types=num_token_types,
            )
            for _ in range(num_device_types)
        ])

        self.spec_heads = nn.ModuleList([
            TokenConditionedDecoder(
                hidden_dim=hidden_dim,
                out_value_dim=spec_out_dim,
                num_token_types=num_token_types,
            )
            for _ in range(num_device_types)
        ])

    def forward(
        self,
        device_latent: torch.Tensor,
        device_type: torch.Tensor,
        param_frequency: torch.Tensor,
        param_type: torch.Tensor,
        spec_frequency: torch.Tensor,
        spec_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device_latent.shape[0] == 0:
            param_out_dim = self.param_heads[0].decoder.mlp[-1].out_features
            spec_out_dim = self.spec_heads[0].decoder.mlp[-1].out_features
            return (
                torch.zeros((0, param_frequency.shape[1], param_out_dim), dtype=torch.float32, device=device_latent.device),
                torch.zeros((0, spec_frequency.shape[1], spec_out_dim), dtype=torch.float32, device=device_latent.device),
            )

        param_preds = []
        spec_preds = []

        for i in range(device_latent.shape[0]):
            d_type = int(device_type[i].item())
            param_pred = self.param_heads[d_type](
                device_latent[i:i + 1],
                param_frequency[i:i + 1],
                param_type[i:i + 1],
            )
            spec_pred = self.spec_heads[d_type](
                device_latent[i:i + 1],
                spec_frequency[i:i + 1],
                spec_type[i:i + 1],
            )
            param_preds.append(param_pred)
            spec_preds.append(spec_pred)

        return torch.cat(param_preds, dim=0), torch.cat(spec_preds, dim=0)