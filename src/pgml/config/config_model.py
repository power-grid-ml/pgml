import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLFlowConfig(BaseModel):
    enabled: bool = False
    experiment_base_name: str = "PowerGridStateEstimation"
    tracking_uri: str = "http://localhost:5000"


class PathsConfig(BaseModel):
    input_dir: Path = Path("./data/input")
    output_dir: Path = Path("./data/output")


class DataLoaderConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 4
    chunk_size_rows: int = 200_000


class PipelineConfig(BaseSettings):
    """
    Root configuration object.
    Resolution order: Default -> YAML (if loaded manually) -> .env file -> Environment Variables.
    """
    paths: PathsConfig = PathsConfig()
    tracking: MLFlowConfig = MLFlowConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()

    # Supports nested env vars (e.g., export PIPELINE__TRACKING__ENABLED=true)
    model_config = SettingsConfigDict(
        env_prefix="PIPELINE__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Loads base parameters from a YAML file, with env vars retaining highest priority."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        return cls.model_validate(yaml_data)
