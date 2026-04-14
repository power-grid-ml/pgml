from pathlib import Path

import lightning as L
import torch
import torch.multiprocessing
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger

from config import resource_dir
from pgml.config import PipelineConfig, config_dir
from pgml.data_pipeline.step_dataloader import get_step_dataloader
from pgml.models.state_estimator import MultiModalStateEstimator
from pgml.training.multitask_engine import MultiTaskStateEstimationEngine, StepWiseCurriculum
from pgml.training.evaluation import (
    LossHistoryPlotter,
    ValidationEvaluator,
    export_training_history_json,
)


def _resolve_rel_abs_path(base_path:Path, target_path:Path) -> Path:
    """
    Checks if target path is relative or absolute.
    If it is a relative path, prepend base_path.
    If path does not exist yet, create it.
    :param base_path:
    :param target_path:
    :return:
    """
    if target_path.is_absolute():
        target_path.mkdir(parents=True, exist_ok=True)
        return target_path
    full_path = Path(base_path) / target_path
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def _infer_model_dims_from_batch(batch) -> dict:
    edge_type = ("node", "physical", "node")
    return {
        "node_static_dim": batch["node"].static_x.shape[1],
        "edge_static_dim": batch[edge_type].static_edge_attr.shape[1],
        "device_static_dim": batch["device"].static_x.shape[1],
        "node_value_dim": batch["node"].meas_value.shape[-1],
        "edge_value_dim": batch["edge"].meas_value.shape[-1],
        "device_param_value_dim": batch["device"].param_value.shape[-1],
        "device_spec_value_dim": batch["device"].spec_value.shape[-1],
    }


def main():
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")

    default_cfg = Path(config_dir) / "default.yaml"
    config = PipelineConfig.from_yaml(default_cfg)
    input_dir = _resolve_rel_abs_path(resource_dir, config.paths.input_dir)
    output_dir = _resolve_rel_abs_path(resource_dir, config.paths.output_dir)

    train_dataset_ids = [2, 3]
    val_dataset_ids = [4]

    train_loader = get_step_dataloader(
        base_data_dir=input_dir,
        dataset_ids=train_dataset_ids,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
    )
    val_loader = get_step_dataloader(
        base_data_dir=input_dir,
        dataset_ids=val_dataset_ids,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
    )

    sample_batch = next(iter(train_loader))
    dims = _infer_model_dims_from_batch(sample_batch)

    model = MultiModalStateEstimator(
        node_static_dim=dims["node_static_dim"],
        edge_static_dim=dims["edge_static_dim"],
        device_static_dim=dims["device_static_dim"],
        node_value_dim=dims["node_value_dim"],
        edge_value_dim=dims["edge_value_dim"],
        device_param_value_dim=dims["device_param_value_dim"],
        device_spec_value_dim=dims["device_spec_value_dim"],
        hidden_dim=64,
    )
    curriculum = StepWiseCurriculum({
        0: {"node_mask_ratio": 0.0, "edge_mask_ratio": 0.0, "device_noise_scale": 0.0, "spectrum_drop_prob": 0.0},
        100: {"node_mask_ratio": 0.1, "edge_mask_ratio": 0.1, "device_noise_scale": 0.01, "spectrum_drop_prob": 0.1},
        200: {"node_mask_ratio": 0.2, "edge_mask_ratio": 0.3, "device_noise_scale": 0.02, "spectrum_drop_prob": 0.2},
    })
    engine = MultiTaskStateEstimationEngine(
        model=model,
        lr=1e-3,
        weight_decay=1e-4,
        alpha_recon=1.0,
        beta_state=1.0,
        curriculum_fn=curriculum
    )

    logger = None
    if config.tracking.enabled:
        logger = MLFlowLogger(
            experiment_name=config.tracking.experiment_base_name,
            tracking_uri=config.tracking.tracking_uri,
        )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss_total",
            mode="min",
            save_top_k=3,
            filename="epoch{epoch:03d}-val_loss_total{val_loss_total:.5f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        default_root_dir=output_dir,
        log_every_n_steps=1,
    )

    trainer.fit(engine, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Export training history collected inside the LightningModule
    history_path = output_dir / "training_history.json"
    export_training_history_json(engine, history_path)

    # Plot loss curves
    plotter = LossHistoryPlotter()
    plotter.plot_from_engine(
        engine=engine,
        output_path=output_dir / "loss_curves.png",
        title="Training and Validation Loss Curves",
    )

    # Full validation summary
    evaluator = ValidationEvaluator(device=engine.device)
    summary_text = evaluator.evaluate(
        engine=engine,
        dataloader=val_loader,
        output_dir=output_dir,
    )

    summary_path = output_dir / "validation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    if logger is not None and logger.experiment is not None:
        run_id = logger.run_id
        try:
            logger.experiment.log_artifact(run_id, str(history_path))
            logger.experiment.log_artifact(run_id, str(output_dir / "loss_curves.png"))
            logger.experiment.log_artifact(run_id, str(summary_path))
        except Exception:
            # TODO: add explicit MLflow artifact upload error handling if needed
            pass

    print(summary_text)


if __name__ == "__main__":
    main()