import torch
import torch.nn as nn


class PowerGridTransformer(nn.Module):
    """
    Global Attention Transformer. Treats nodes as tokens in a sequence.
    Requires external injection of graph distances (e.g., Shortest Path Encodings)
    to retain physical topology awareness.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_nodes: int, num_layers: int = 4,
                 heads: int = 4):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        # Spatial Encoding: Learnable embedding based on node IDs or Laplacian eigenvectors
        # This replaces the positional encoding used in NLP.
        self.spatial_embedding = nn.Embedding(num_nodes, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_fused: torch.Tensor, node_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        # 1. Encode features
        x = self.node_encoder(x_fused)

        # 2. Add structural encodings
        x = x + self.spatial_embedding(node_ids)

        # 3. Reshape from[total_nodes, hidden_dim] to [batch_size, num_nodes_per_graph, hidden_dim]
        # Transformers require a batched sequence format, unlike PyG's flat graph batching
        num_nodes_per_graph = x.shape[0] // batch_size
        x_seq = x.view(batch_size, num_nodes_per_graph, -1)

        # 4. Global Dense Attention (All nodes attend to all nodes)
        out_seq = self.transformer(x_seq)

        # 5. Decode back to physical states
        out_flat = out_seq.view(-1, out_seq.shape[-1])
        return self.decoder(out_flat)