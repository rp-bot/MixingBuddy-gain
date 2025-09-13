"""
Experiment tracking utilities for Weights & Biases and MLflow.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import wandb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracking interface for multiple backends."""

    def __init__(self, config: DictConfig, backend: str = "wandb"):
        """
        Initialize experiment tracker.

        Args:
            config: Hydra configuration object
            backend: Tracking backend ("wandb" or "mlflow")
        """
        self.config = config
        self.backend = backend
        self.run = None

        if backend == "wandb":
            self._init_wandb()
        elif backend == "mlflow":
            self._init_mlflow()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            # Set run name if not provided
            run_name = self.config.experiment_tracking.get("name")
            if not run_name:
                run_name = f"lora-{self.config.model.pretrained_model_name_or_path.split('/')[-1]}-{self.config.env.seed}"

            # Initialize wandb
            self.run = wandb.init(
                project=self.config.experiment_tracking.project,
                entity=self.config.experiment_tracking.get("entity"),
                name=run_name,
                tags=self.config.experiment_tracking.get("tags", []),
                notes=self.config.experiment_tracking.get("notes", ""),
                reinit=self.config.experiment_tracking.get("reinit", True),
                save_code=self.config.experiment_tracking.get("save_code", True),
                config=self._flatten_config(self.config),
            )
            logger.info(f"Initialized Weights & Biases run: {self.run.name}")

        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")
            self.run = None

    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            # Set run name if not provided
            run_name = self.config.experiment_tracking.get("run_name")
            if not run_name:
                run_name = f"lora-{self.config.model.pretrained_model_name_or_path.split('/')[-1]}-{self.config.env.seed}"

            # Set experiment name
            experiment_name = self.config.experiment_tracking.get(
                "experiment_name", "llm-lora-automatic-mixing"
            )
            mlflow.set_experiment(experiment_name)

            # Start run
            self.run = mlflow.start_run(
                run_name=run_name, tags=self.config.experiment_tracking.get("tags", {})
            )

            # Log parameters
            self.log_params(self._flatten_config(self.config))
            logger.info(f"Initialized MLflow run: {self.run.info.run_id}")

        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.run = None

    def _flatten_config(
        self, config: DictConfig, parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested configuration for logging."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, DictConfig):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the tracking backend."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.config.update(params)
            elif self.backend == "mlflow":
                for key, value in params.items():
                    mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ):
        """Log metrics to the tracking backend."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.log(metrics, step=step)
            elif self.backend == "mlflow":
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts (files/directories) to the tracking backend."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.log_artifact(local_dir, name=artifact_path or "artifacts")
            elif self.backend == "mlflow":
                mlflow.log_artifacts(local_dir, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")

    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model to the tracking backend."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.log_model(model_path, name=model_name)
            elif self.backend == "mlflow":
                mlflow.log_model(model_path, model_name)
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_plots(self, plots: Dict[str, Any]):
        """Log plots/figures to the tracking backend."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.log(plots)
            elif self.backend == "mlflow":
                for name, plot in plots.items():
                    mlflow.log_figure(plot, f"{name}.png")
        except Exception as e:
            logger.error(f"Failed to log plots: {e}")

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch model for gradients/parameters (WandB only)."""
        if self.backend == "wandb" and self.run:
            try:
                wandb.watch(model, log=log, log_freq=log_freq)
            except Exception as e:
                logger.error(f"Failed to watch model: {e}")

    def finish(self):
        """Finish the experiment run."""
        if not self.run:
            return

        try:
            if self.backend == "wandb":
                wandb.finish()
            elif self.backend == "mlflow":
                mlflow.end_run()
            logger.info("Experiment tracking finished")
        except Exception as e:
            logger.error(f"Failed to finish experiment tracking: {e}")

    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if not self.run:
            return None

        if self.backend == "wandb":
            return self.run.id
        elif self.backend == "mlflow":
            return self.run.info.run_id
        return None

    def get_run_url(self) -> Optional[str]:
        """Get the URL to view the current run."""
        if not self.run:
            return None

        if self.backend == "wandb":
            return self.run.url
        elif self.backend == "mlflow":
            return mlflow.get_tracking_uri()
        return None


def setup_logging(config: DictConfig) -> logging.Logger:
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(config.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "training.log"),
        ],
    )

    return logging.getLogger(__name__)


def save_experiment_config(config: DictConfig, output_dir: Path):
    """Save experiment configuration to file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as YAML
    with open(output_dir / "config.yaml", "w") as f:
        from omegaconf import OmegaConf

        OmegaConf.save(config, f)

    # Save as JSON for easy reading
    with open(output_dir / "config.json", "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)

    logger.info(f"Saved experiment configuration to {output_dir}")
