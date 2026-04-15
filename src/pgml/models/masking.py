from __future__ import annotations

import torch
import torch.nn as nn


class LatentObservabilityMasker(nn.Module):
    """
    Applies curriculum-controlled masking to node and edge latent measurements.

    Behavior:
    - a fraction of node/edge measurement latents is replaced by a learnable mask token
    - an observability indicator is returned for downstream conditioning

    Inputs:
    - latent: [N, H]
    - mask_ratio: float in [0, 1]

    Outputs:
    - masked_latent: [N, H]
    - indicator:     [N, 1]   1.0 = observed, 0.0 = masked
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        latent: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = latent.shape[0]
        device = latent.device

        if n == 0:
            return (
                latent,
                torch.zeros((0, 1), dtype=torch.float32, device=device),
            )

        if mask_ratio <= 0.0:
            indicator = torch.ones((n, 1), dtype=torch.float32, device=device)
            return latent, indicator

        is_observed = (torch.rand(n, device=device) > mask_ratio)
        indicator = is_observed.unsqueeze(-1).to(torch.float32)

        masked_latent = torch.where(
            is_observed.unsqueeze(-1),
            latent,
            self.mask_token.to(latent.dtype).expand(n, -1),
        )

        return masked_latent, indicator


class DeviceInputNoiser(nn.Module):
    """
    Applies curriculum-controlled corruption to device parameter and spectrum tokens.

    Mechanisms:
    - additive Gaussian noise on token values
    - optional full-spectrum replacement by a learned unknown-spectrum token

    This simulates imperfect pseudo-measurements and missing harmonic priors.

    #TODO: Add device-type-specific noise models if later experiments show
    #      generators, loads, and vsources require different corruption statistics.
    """
    def __init__(self, param_value_dim: int, spec_value_dim: int):
        super().__init__()
        self.param_value_dim = param_value_dim
        self.spec_value_dim = spec_value_dim

        self.unknown_spectrum_token = nn.Parameter(torch.zeros(1, 1, spec_value_dim))
        nn.init.normal_(self.unknown_spectrum_token, std=0.02)

    def forward(
        self,
        param_value: torch.Tensor,
        param_mask: torch.Tensor,
        spec_value: torch.Tensor,
        spec_mask: torch.Tensor,
        noise_scale: float,
        spectrum_drop_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if noise_scale <= 0.0 and spectrum_drop_prob <= 0.0:
            return param_value, spec_value

        param_out = param_value.clone()
        spec_out = spec_value.clone()

        if param_out.numel() > 0 and noise_scale > 0.0:
            param_noise = torch.randn_like(param_out) * noise_scale
            param_out = param_out + param_noise * param_mask.unsqueeze(-1).to(param_out.dtype)

        if spec_out.numel() > 0 and spectrum_drop_prob > 0.0 and spec_out.shape[0] > 0:
            device_drop = torch.rand(spec_out.shape[0], device=spec_out.device) < spectrum_drop_prob
            if device_drop.any():
                spec_out[device_drop] = self.unknown_spectrum_token.to(spec_out.dtype).expand(
                    int(device_drop.sum().item()),
                    spec_out.shape[1],
                    -1
                )

        return param_out, spec_out
