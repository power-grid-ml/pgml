import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from pgml.config import PipelineConfig
from data_pipeline import get_train_dataloader
from models.gnn import PowerGridGNN
from models.masking import ObservabilityMasker
from training.engine import StateEstimationEngine


def main():
    # 1. Load dynamic configuration
    config = PipelineConfig()  # Pulls from .env automatically

    # 2. Dimensions (Example values, derived from earlier scaler logic)
    static_dim = 14  # TopologyExporter node outputs
    dynamic_dim = 6  # V1_real, V1_imag ... V3_real, V3_imag
    edge_dim = 25  # TopologyExporter edge outputs
    hidden_dim = 128

    # 3. Model & Masker
    masker = ObservabilityMasker(dynamic_feature_dim=dynamic_dim)

    # Fused input dim: static + dynamic + 1 (indicator boolean)
    fused_in_dim = static_dim + dynamic_dim + 1

    model = PowerGridGNN(
        input_dim=fused_in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        output_dim=dynamic_dim
    )

    # 4. Lightning Engine
    engine = StateEstimationEngine(
        model=model,
        masker=masker,
        config=config,
        dynamic_feature_dim=dynamic_dim
    )

    # 5. DataLoaders (Utilizing the Iterable PyArrow pipeline)
    train_loader = get_train_dataloader(
        base_data_dir=config.paths.input_dir,
        train_dataset_ids=[2, 3, 4],
        stats_file=config.paths.input_dir / "scaling_stats.json",
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers
    )

    # 6. MLFlow Logger setup
    logger = None
    if config.tracking.enabled:
        logger = MLFlowLogger(
            experiment_name=config.tracking.experiment_base_name,
            tracking_uri=config.tracking.tracking_uri,
        )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        LearningRateMonitor(logging_interval='step')
    ]

    # 7. Trainer
    trainer = L.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed"  # Crucial for scaling TransformerConvs on modern GPUs
    )

    trainer.fit(engine, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()