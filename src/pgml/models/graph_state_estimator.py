from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphStateEstimator(nn.Module):
    """
    Lightweight edge-aware GNN operating on node and edge latent features.

    Input:
    - node_latent: [num_nodes, hidden_dim]
    - edge_index:  [2, num_edges]
    - edge_latent: [num_edges, hidden_dim]

    Output:
    - updated_node_latent: [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        out_channels = max(hidden_dim // heads, 1)

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    heads=heads,
                    edge_dim=hidden_dim,
                    beta=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_latent: torch.Tensor,
    ) -> torch.Tensor:
        x = node_latent
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index, edge_latent)
            x = norm(x + x_res)
            x = torch.nn.functional.gelu(x)
        return x