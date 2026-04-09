# training/engine.py
import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.masking import ObservabilityMasker
from pgml.config import PipelineConfig


class StateEstimationEngine(L.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            masker: ObservabilityMasker,
            config: PipelineConfig,
            dynamic_feature_dim: int
    ):
        super().__init__()
        self.model = model
        self.masker = masker
        self.config = config
        self.dynamic_feature_dim = dynamic_feature_dim

        # Curriculum Learning state
        self.current_mask_ratio = 0.0

        self.save_hyperparameters(ignore=['model', 'masker'])

    def forward(self, batch) -> torch.Tensor:
        # Extract features from the HeteroData PyG batch object
        static_x = batch['node'].static_x
        dynamic_x = batch['node'].x[:, -self.dynamic_feature_dim:]

        edge_type = ('node', 'physical', 'node')
        edge_index = batch[edge_type].edge_index
        edge_attr = batch[edge_type].static_edge_attr

        # Apply observability masking
        x_fused = self.masker(static_x, dynamic_x, mask_ratio=self.current_mask_ratio)

        # Predict the full dynamic state (reconstruction)
        return self.model(x_fused, edge_index, edge_attr)

    def _shared_step(self, batch, batch_idx, phase: str):
        dynamic_x_target = batch['node'].x[:, -self.dynamic_feature_dim:]

        # Forward pass
        predictions = self(batch)

        # Supervised Loss (MSE) - Computed over ALL nodes, enforcing state estimation
        loss = F.mse_loss(predictions, dynamic_x_target)

        # Separate evaluation: error strictly on unmeasured nodes
        if phase != "train" and self.current_mask_ratio > 0.0:
            # The indicator column is the last column of the fused input
            # We reconstruct it here to find unmeasured indices
            is_measured = batch['node'].x[:, -1] if hasattr(batch['node'], 'x_fused') else None
            # Future extension: Log separate metrics for unmeasured nodes here

        self.log(f"{phase}_loss", loss, batch_size=batch.num_graphs, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        # Validation is usually evaluated at the target sparsity (e.g., 95% unmeasured)
        original_mask_ratio = self.current_mask_ratio
        self.current_mask_ratio = 0.95

        loss = self._shared_step(batch, batch_idx, "val")

        self.current_mask_ratio = original_mask_ratio
        return loss

    def on_train_epoch_start(self):
        """
        Updates the Curriculum Learning mask ratio.
        Linearly increases sparsity from 0% to 95% over the first 50 epochs.
        """
        max_epochs_curriculum = 50
        max_mask_ratio = 0.95

        if self.current_epoch < max_epochs_curriculum:
            self.current_mask_ratio = (self.current_epoch / max_epochs_curriculum) * max_mask_ratio
        else:
            self.current_mask_ratio = max_mask_ratio

        self.log("mask_ratio", self.current_mask_ratio)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Cosine annealing with warm restarts prevents convergence into poor local minima
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }