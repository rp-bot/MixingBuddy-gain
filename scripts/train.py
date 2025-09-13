#!/usr/bin/env python3
"""
Training script for LoRA fine-tuning.
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from data.dataset import DataProcessor, create_sample_data
from models.lora_model import create_lora_model
from training.trainer import LoRATrainer
from utils.experiment_tracking import (
    ExperimentTracker,
    save_experiment_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Set up logging
    setup_logging(config)
    logger.info("Starting LoRA fine-tuning...")

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Create output directories
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    save_experiment_config(config, output_dir)

    # Set random seeds for reproducibility
    import random

    import numpy as np
    import torch

    torch.manual_seed(config.env.seed)
    np.random.seed(config.env.seed)
    random.seed(config.env.seed)

    # Set device
    if config.env.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.env.device

    logger.info(f"Using device: {device}")

    # Create sample data if it doesn't exist
    data_dir = Path(config.paths.data_dir)
    if not (data_dir / "processed").exists():
        logger.info("Creating sample data...")
        create_sample_data(data_dir / "processed", num_samples=1000)

    # Initialize experiment tracking
    experiment_tracker = None
    if config.get("experiment_tracking"):
        try:
            backend = "wandb" if "wandb" in config.experiment_tracking else "mlflow"
            experiment_tracker = ExperimentTracker(config, backend=backend)
            logger.info(f"Initialized experiment tracking with {backend}")
        except Exception as e:
            logger.warning(f"Failed to initialize experiment tracking: {e}")

    try:
        # Create model
        logger.info("Creating LoRA model...")
        model = create_lora_model(config.model)

        # Log model info
        param_info = model.get_trainable_parameters()
        logger.info(f"Model parameters: {param_info}")

        if experiment_tracker:
            experiment_tracker.log_metrics(
                {
                    "total_parameters": param_info["total"],
                    "trainable_parameters": param_info["trainable"],
                    "trainable_percentage": param_info["percentage"],
                }
            )

        # Create data processor
        logger.info("Setting up data processing...")
        data_processor = DataProcessor(config.data)

        # Load datasets
        train_dataset = data_processor.load_dataset("train")
        val_dataset = data_processor.load_dataset("validation")

        # Create data loaders
        train_dataloader = data_processor.create_dataloader(train_dataset, "train")
        val_dataloader = data_processor.create_dataloader(val_dataset, "validation")

        # Log dataset statistics
        train_stats = data_processor.get_dataset_stats(train_dataset)
        val_stats = data_processor.get_dataset_stats(val_dataset)

        logger.info(f"Train dataset stats: {train_stats}")
        logger.info(f"Validation dataset stats: {val_stats}")

        if experiment_tracker:
            experiment_tracker.log_metrics(
                {
                    "train_samples": train_stats["num_samples"],
                    "val_samples": val_stats["num_samples"],
                    "train_avg_length": train_stats["avg_length"],
                    "val_avg_length": val_stats["avg_length"],
                }
            )

        # Create trainer
        logger.info("Setting up trainer...")
        trainer = LoRATrainer(model, config, experiment_tracker)

        # Train the model
        logger.info("Starting training...")
        trainer.train(train_dataloader, val_dataloader)

        # Evaluate the model
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate(val_dataloader)

        # Save the final model
        logger.info("Saving model...")
        trainer.save_model()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        # Finish experiment tracking
        if experiment_tracker:
            experiment_tracker.finish()


if __name__ == "__main__":
    main()
