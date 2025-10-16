"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import LoRATrainer  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    initialize_tokenizer,
    load_dataset,
    initialize_experiment_tracker
)


def initialize_model(cfg: DictConfig):
    """Initialize model with LoRA/QLoRA setup."""
    print("Initializing model...")

    # Create LoRA configuration
    lora_config = create_lora_config(cfg)

    # Load tokenizer
    tokenizer = initialize_tokenizer(cfg.model.model_name)

    # Initialize model based on configuration
    if cfg.model.use_qlora:
        llm = initialize_qlora_model(cfg, lora_config, tokenizer)
    else:
        llm = initialize_lora_model(cfg, lora_config, tokenizer)

    # Initialize the multimodal model with the correctly configured LLM
    model = ModularMultimodalModel(
        model_name=cfg.model.model_name,
        use_qlora=cfg.model.use_qlora,
        lora_config=lora_config,
        llm=llm,  # Pass the pre-configured LLM
        tokenizer=tokenizer,  # Pass the tokenizer
        encoder_config=cfg.model.get("encoder"),  # Pass encoder configuration
    )

    print("Model initialized.")
    model.print_trainable_parameters()
    return model


def load_datasets(cfg: DictConfig, model):
    """Load training, validation, and test datasets."""
    import torch
    from torch.utils.data import random_split

    print("Loading data...")

    # Load full training dataset
    full_train_dataset = load_dataset(cfg, model, "train")

    # Load test dataset
    test_dataset = load_dataset(cfg, model, "test")

    # Split training data into train/validation (80/20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.env.seed),  # Reproducible split
    )

    print("Dataset sizes:")
    print(f"  - Training: {len(train_dataset)} samples")
    print(f"  - Validation: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    print("Data loaded.")
    return train_dataset, val_dataset, test_dataset


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function.
    """
    # --- 1. Initialize Experiment Tracker ---
    tracker = initialize_experiment_tracker(cfg)

    # --- 2. Initialize Model ---
    model = initialize_model(cfg)

    # --- 3. Load Data ---
    train_dataset, val_dataset, test_dataset = load_datasets(cfg, model)

    # --- 4. Initialize Trainer ---
    print("Initializing trainer...")
    trainer = LoRATrainer(
        model=model,
        config=cfg,
        experiment_tracker=tracker,
    )
    print("Trainer initialized.")

    # --- 5. Train ---
    print("Starting training...")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print("Training progress will be logged to Wandb...")

    trainer.train(train_dataset=train_dataset, eval_dataset=val_dataset)
    print("Training finished.")

    # --- 6. Save Model ---
    print("Saving model...")
    trainer.save_model()
    print("Model saved.")

    # --- 7. Finish Experiment Tracking ---
    tracker.finish()


if __name__ == "__main__":
    main()
