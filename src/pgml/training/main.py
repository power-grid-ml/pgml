import json
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from config import resource_dir
from data_pipeline.scaler import TorchStandardScaler
from pgml.config import PipelineConfig, config_dir
from data_pipeline import get_train_dataloader, TopologyCache
from models.gnn import PowerGridGNN
from models.masking import ObservabilityMasker
from training.engine import StateEstimationEngine


def main():
    # 1. Load dynamic configuration
    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)

    target_features =[
        "v1_real", "v1_imag",
        "v2_real", "v2_imag",
        "v3_real", "v3_imag"
    ]
    # 2. Scaler
    scaler = TorchStandardScaler(dim=1, feature_names=target_features)

    stats_file = Path(resource_dir) / config.paths.input_dir / "scaling_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Scaling stats not found at {stats_file}. Run stats_compiler.py.")

    with open(stats_file, 'r', encoding='utf-8') as f:
        all_stats = json.load(f)

    # Populate the scaler's PyTorch buffers out-of-core
    for freq_str, freq_stats in all_stats.get("node_data", {}).items():
        scaler.load_from_stats(stats_dict=freq_stats, group_key=freq_str)

    # 3. Dynamic Dimensions
    train_dataset_ids = [2, 3]

    # Initialize TopologyCache to peek at the graph structure
    topo_cache = TopologyCache(resource_dir / config.paths.input_dir)

    # Read metadata of the first dataset to find its topology_id
    with open(resource_dir / config.paths.input_dir / f"dataset_{train_dataset_ids[0]}" / "metadata.json", "r") as f:
        meta = json.load(f)

    # Load the base HeteroData object for this topology
    sample_topo = topo_cache.get_topology(meta["topology_id"])

    # FIX: Use the tuple key to infer edge dimensions
    edge_type = ('node', 'physical', 'node')

    static_dim = sample_topo['node'].static_x.shape[1]
    edge_dim = sample_topo[edge_type].static_edge_attr.shape[1]
    dynamic_dim = len(target_features)

    print(f"Inferred Dimensions - Static Node: {static_dim}, Edge: {edge_dim}, Dynamic Node: {dynamic_dim}")

    fused_in_dim = static_dim + dynamic_dim + 1  # +1 for Observability Mask Indicator

    # 4. Model & Masker
    masker = ObservabilityMasker(dynamic_feature_dim=dynamic_dim)
    model = PowerGridGNN(
        input_dim=fused_in_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        output_dim=dynamic_dim
    )

    engine = StateEstimationEngine(
        model=model,
        masker=masker,
        config=config,
        dynamic_feature_dim=dynamic_dim
    )

    # 5. DataLoaders (Utilizing the Iterable PyArrow pipeline)
    train_loader = get_train_dataloader(
        base_data_dir=Path(resource_dir) / config.paths.input_dir,
        train_dataset_ids=train_dataset_ids,
        scaler=scaler,
        feature_prefixes=["v1", "v2", "v3"],
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
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