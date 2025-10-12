"""
Experiment tracking utilities for Weights & Biases and MLflow.
"""

import json
import logging

# import os
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
        self._current_run_name = None  # Initialize to ensure it exists

        if backend == "wandb":
            self._init_wandb()
        elif backend == "mlflow":
            self._init_mlflow()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Ensure _current_run_name is always set
        if not self._current_run_name:
            self._current_run_name = self._generate_run_name()

    def _generate_run_name(self):
        """Generate run name using naming convention."""
        # Check if custom name is provided
        custom_name = self.config.experiment_tracking.get("name")
        if custom_name:
            return custom_name

        # Extract components for naming convention
        model_name = self.config.model.model_name

        # Get model abbreviation from config mapping
        model_abbr = self.config.experiment_naming.naming.components.model_abbr.get(
            model_name, model_name.lower().replace("-instruct", "").replace("-", "")
        )

        # LoRA configuration
        lora_config = self.config.model.lora
        rank = lora_config.r
        alpha = lora_config.lora_alpha
        lora_str = f"r{rank}a{alpha}"

        # Dataset identifier from config mapping
        dataset_path = self.config.data.train_jsonl_path
        if "musdb" in dataset_path.lower():
            dataset_abbr = "musdb"
        else:
            dataset_abbr = "custom"

        # Experiment type from config mapping
        if self.config.model.use_qlora:
            exp_type = self.config.experiment_naming.naming.components.exp_type.qlora
        else:
            exp_type = self.config.experiment_naming.naming.components.exp_type.lora

        # Version (use timestamp for uniqueness)
        import datetime

        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")

        # Construct run name: {exp_type}-{model_abbr}-{lora_config}-{dataset_abbr}-{timestamp}
        run_name = f"{exp_type}-{model_abbr}-{lora_str}-{dataset_abbr}-{timestamp}"

        return run_name

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            # Generate run name using naming convention
            run_name = self._generate_run_name()
            self._current_run_name = run_name  # Store for trainer access

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
            # Generate run name using naming convention
            run_name = self._generate_run_name()
            self._current_run_name = run_name  # Store for trainer access

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
                # Convert non-serializable values to strings
                serialized_value = self._serialize_value(v)
                items.append((new_key, serialized_value))
        return dict(items)

    def _serialize_value(self, value: Any) -> Union[str, int, float, bool, None]:
        """Convert value to a format that can be logged to Wandb."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to strings
            return str(value)
        elif isinstance(value, dict):
            # Convert dicts to strings
            return str(value)
        else:
            # Convert any other type to string
            return str(value)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the tracking backend."""
        if not self.run:
            return

        try:
            # Filter out any remaining non-serializable values
            serialized_params = {}
            for key, value in params.items():
                serialized_value = self._serialize_value(value)
                serialized_params[key] = serialized_value

            if self.backend == "wandb":
                wandb.config.update(serialized_params)
            elif self.backend == "mlflow":
                for key, value in serialized_params.items():
                    mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            # Log the problematic parameters for debugging
            logger.error(
                f"Problematic parameters: {[(k, type(v), v) for k, v in params.items() if not isinstance(v, (str, int, float, bool, type(None)))]}"
            )

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
