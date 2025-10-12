"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import warnings
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MixingDataset  # noqa: E402
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import LoRATrainer  # noqa: E402
from src.utils.experiment_tracking import ExperimentTracker  # noqa: E402

# Suppress warnings
# The "resume_download" warning is a FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)
# The "special tokens" warning is logged by the transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)


def create_lora_config(cfg: DictConfig):
    """Create LoRA configuration from config."""
    from peft import LoraConfig, TaskType

    # Map string task type to TaskType enum
    task_type_mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
    }

    task_type = task_type_mapping.get(cfg.model.lora.task_type, TaskType.CAUSAL_LM)

    return LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        target_modules=cfg.model.lora.target_modules,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias=cfg.model.lora.bias,
        task_type=task_type,
    )


def setup_qlora_model(model, cfg: DictConfig, lora_config):
    """Setup QLoRA with quantization and LoRA."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM

    print("Setting up QLoRA...")

    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(
            torch, cfg.model.quantization.bnb_4bit_compute_dtype
        ),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # Reload model with quantization
    model.llm = AutoModelForCausalLM.from_pretrained(
        model.model_name,
        torch_dtype="auto",
        quantization_config=quantization_config,
    )

    # Prepare for k-bit training
    model.llm = prepare_model_for_kbit_training(model.llm)
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    model.llm = get_peft_model(model.llm, lora_config)
    print("QLoRA setup complete.")


def setup_standard_lora(model, lora_config):
    """Setup standard LoRA."""
    from peft import get_peft_model

    print("Using standard LoRA...")
    model.llm = get_peft_model(model.llm, lora_config)
    print("Standard LoRA setup complete.")


def initialize_model(cfg: DictConfig):
    """Initialize model with LoRA/QLoRA setup."""
    print("Initializing model...")

    # Create LoRA configuration
    lora_config = create_lora_config(cfg)

    # Initialize basic model
    model = ModularMultimodalModel(
        model_name=cfg.model.model_name,
        use_qlora=False,
        lora_config=None,
    )

    # Apply LoRA setup based on config
    if cfg.model.use_qlora:
        setup_qlora_model(model, cfg, lora_config)
    else:
        setup_standard_lora(model, lora_config)

    print("Model initialized.")
    model.print_trainable_parameters()
    return model


def initialize_experiment_tracker(cfg: DictConfig):
    """Initialize experiment tracker."""
    print("Initializing experiment tracker...")

    if cfg.experiment_tracking.get("use_wandb", False):
        tracker = ExperimentTracker(config=cfg, backend="wandb")
    elif cfg.experiment_tracking.get("use_mlflow", False):
        tracker = ExperimentTracker(config=cfg, backend="mlflow")
    else:
        raise ValueError("No experiment tracking backend configured")

    print("Experiment tracker initialized.")
    return tracker


def load_datasets(cfg: DictConfig, model):
    """Load training, validation, and test datasets."""
    import torch
    from torch.utils.data import random_split

    print("Loading data...")

    # Load full training dataset
    full_train_dataset = MixingDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        tokenizer=model.tokenizer,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.get("limit"),
    )

    # Load test dataset (renamed from eval)
    test_dataset = MixingDataset(
        jsonl_path=cfg.data.test_jsonl_path,  # Renamed from eval_jsonl_path
        audio_root=cfg.data.test_audio_root,  # Renamed from eval_audio_root
        tokenizer=model.tokenizer,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.get("limit"),
    )

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

    # Quick validation
    sample = train_dataset[0]
    print(f"Sample audio shape: {sample['audio'].shape}")
    print("Data loaded.")

    return train_dataset, val_dataset, test_dataset


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """
    Main training function.
    """
    # --- 1. Initialize Model ---
    model = initialize_model(cfg)

    # --- 2. Initialize Experiment Tracker ---
    tracker = initialize_experiment_tracker(cfg)

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

    # --- 6. Final Test Evaluation ---
    print("Running final test evaluation...")

    # Aggressive memory cleanup before evaluation
    import torch
    import gc

    # Clear training-related variables
    del train_dataset, val_dataset
    gc.collect()

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Force garbage collection again
    gc.collect()

    print(f"GPU memory before evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        # Try evaluation on full test dataset with cleaned memory
        test_results = trainer.evaluate(test_dataset)
        print(f"Final test results: {test_results}")

        # Log final test results
        tracker.log_metrics(test_results, step="final")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Skipping final evaluation.")
        test_results = {"eval_loss": "N/A", "eval_perplexity": "N/A"}

    # --- 7. Save Model ---
    print("Saving model...")
    trainer.save_model()
    print("Model saved.")

    # --- 8. Finish Experiment Tracking ---
    tracker.finish()


if __name__ == "__main__":
    main()
