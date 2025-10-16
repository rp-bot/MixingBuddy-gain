"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import torch
from torch.utils.data import random_split
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collator import MultimodalDataCollator  # noqa: E402
from src.data.dataset import MixingDataset  # noqa: E402
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import ExperimentTrackingCallback  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    initialize_tokenizer,
    initialize_experiment_tracker,
)


def initialize_model_and_tokenizer(cfg: DictConfig):
    """Initialize model, tokenizer, and LoRA configuration."""
    print("Initializing model and tokenizer...")
    tokenizer = initialize_tokenizer(cfg.model.model_name)
    lora_config = create_lora_config(cfg)

    if cfg.model.use_qlora:
        llm = initialize_qlora_model(cfg, lora_config, tokenizer)
    else:
        llm = initialize_lora_model(cfg, lora_config, tokenizer)

    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),
    )
    print("Model and tokenizer initialized.")
    model.print_trainable_parameters()
    return model, tokenizer


def load_datasets(cfg: DictConfig, tokenizer):
    """Load and split train, validation, and test datasets."""
    print("Loading data...")
    full_train_dataset = MixingDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )
    test_dataset = MixingDataset(
        jsonl_path=cfg.data.test_jsonl_path,
        audio_root=cfg.data.test_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.env.seed),
    )

    print(
        f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    tracker = initialize_experiment_tracker(cfg)
    model, tokenizer = initialize_model_and_tokenizer(cfg)
    train_dataset, val_dataset, test_dataset = load_datasets(cfg, tokenizer)

    # Convert Hydra config to a dictionary to safely modify it
    training_args_dict = OmegaConf.to_container(
        cfg.training.training_args, resolve=True
    )
    # Prevent SFTTrainer from removing custom columns like 'audio' and 'messages'
    training_args_dict["remove_unused_columns"] = False
    training_args = TrainingArguments(**training_args_dict)

    callbacks = [ExperimentTrackingCallback(tracker, model)]
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping.patience
            )
        )

    data_collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # dataset_text_field is removed as the collator now handles all formatting.
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    print("Saving model...")
    trainer.save_model(training_args.output_dir)
    print("Model saved.")

    tracker.finish()


if __name__ == "__main__":
    main()
