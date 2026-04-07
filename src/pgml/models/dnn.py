# models/dense.py
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class BaselineNodeMLP(nn.Module):
    """
    Baseline dense network that operates independently on each node.
    Cannot route power flow information across lines.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_fused: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Edge information is explicitly ignored in the baseline
        return self.net(x_fused)