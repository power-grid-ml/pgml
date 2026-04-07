import torch
import torch.nn as nn


class ObservabilityMasker(nn.Module):
    """
    Applies a dropout-like mask to dynamic node features to simulate unmeasured grid nodes.
    Replaces masked node features with a learnable [MASK] token and appends a boolean
    indicator column to the feature matrix so the network knows which data is synthetic.
    """

    def __init__(self, dynamic_feature_dim: int):
        super().__init__()
        self.dynamic_feature_dim = dynamic_feature_dim
        # Learnable token representing an "unknown" measurement
        self.mask_token = nn.Parameter(torch.zeros(1, dynamic_feature_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, static_x: torch.Tensor, dynamic_x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Args:
            static_x: Tensor of shape[num_nodes, static_dim] (known grid parameters)
            dynamic_x: Tensor of shape[num_nodes, dynamic_dim] (measurements)
            mask_ratio: Probability of a node being unmeasured (0.0 to 1.0)

        Returns:
            Fused tensor of shape[num_nodes, static_dim + dynamic_dim + 1]
        """
        num_nodes = dynamic_x.size(0)
        device = dynamic_x.device

        if mask_ratio <= 0.0:
            # 1.0 indicates "measured"
            indicator = torch.ones((num_nodes, 1), device=device, dtype=torch.float32)
            masked_dynamic = dynamic_x
        else:
            # Generate random mask for this batch
            is_measured = torch.rand(num_nodes, device=device) > mask_ratio
            indicator = is_measured.unsqueeze(1).to(torch.float32)

            # Broadcast mask token and apply
            masked_dynamic = torch.where(
                is_measured.unsqueeze(1),
                dynamic_x,
                self.mask_token.expand(num_nodes, -1)
            )

        # Concatenate: [Static Features, Masked Dynamic Features, Measurement Indicator]
        return torch.cat([static_x, masked_dynamic, indicator], dim=-1)