import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class PowerGridGNN(nn.Module):
    """
    Graph Neural Network utilizing edge-conditioned Transformer Convolutions.
    Routes state estimations through the physical connections of the grid.
    """

    def __init__(
            self,
            input_dim: int,
            edge_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 4,
            heads: int = 4
    ):
        super().__init__()

        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # TransformerConv uses edge attributes to compute attention weights
            conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=hidden_dim,
                beta=True,  # Adds a skip connection mechanism via gating
                dropout=0.1
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_fused: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Encode inputs to latent space
        x = self.node_encoder(x_fused)
        edge_attr_enc = self.edge_encoder(edge_attr)

        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            # Residual connection
            x_res = x
            x = conv(x, edge_index, edge_attr_enc)
            x = norm(x + x_res)
            x = torch.nn.functional.gelu(x)

        # Decode back to physical state dimensions
        return self.decoder(x)