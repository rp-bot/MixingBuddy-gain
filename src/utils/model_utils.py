"""
Shared utilities for model initialization and configuration.
"""

import warnings
import logging
from typing import Optional
from omegaconf import DictConfig

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.dataset import MixingDataset
from src.utils.experiment_tracking import ExperimentTracker

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


def initialize_tokenizer(model_name: str):
    """Initializes and configures the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_lora_config(cfg: DictConfig):
    """Create LoRA configuration from config."""
    from peft import LoraConfig, TaskType
    from omegaconf import OmegaConf

    # Map string task type to TaskType enum
    task_type_mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
    }

    task_type = task_type_mapping.get(cfg.model.lora.task_type, TaskType.CAUSAL_LM)

    # Convert OmegaConf objects to regular Python objects to avoid serialization issues
    target_modules = OmegaConf.to_container(cfg.model.lora.target_modules, resolve=True)

    return LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias=cfg.model.lora.bias,
        task_type=task_type,
    )


def initialize_lora_model(cfg: DictConfig, lora_config, tokenizer):
    """Initialize model with standard LoRA (no quantization)."""
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    print("Loading model with standard LoRA...")

    # Load model without quantization
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype="auto",
    )

    # Apply LoRA
    llm = get_peft_model(llm, lora_config)
    print("Standard LoRA setup complete.")

    return llm


def initialize_qlora_model(cfg: DictConfig, lora_config, tokenizer):
    """Initialize model with QLoRA quantization."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM

    print("Loading model with QLoRA quantization...")

    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(
            torch, cfg.model.quantization.bnb_4bit_compute_dtype
        ),
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # Load model with quantization
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype="auto",
        quantization_config=quantization_config,
    )

    # Prepare for k-bit training
    llm = prepare_model_for_kbit_training(llm)
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA
    llm = get_peft_model(llm, lora_config)
    print("QLoRA setup complete.")

    return llm


def find_latest_checkpoint(
    base_dir: str = "outputs/checkpoints/mixing_buddy_milestone_0",
) -> Optional[str]:
    """Find the latest checkpoint directory."""
    from pathlib import Path

    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    # Find all checkpoint directories
    checkpoint_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not checkpoint_dirs:
        return None

    # Sort by modification time (newest first)
    latest_dir = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)

    print(f"Found latest checkpoint: {latest_dir}")
    return str(latest_dir)


def initialize_experiment_tracker(cfg: DictConfig, required: bool = True):
    """Initialize experiment tracker."""
    from src.utils.experiment_tracking import ExperimentTracker

    print("Initializing experiment tracker...")

    if cfg.experiment_tracking.get("use_wandb", False):
        tracker = ExperimentTracker(config=cfg, backend="wandb")
    elif cfg.experiment_tracking.get("use_mlflow", False):
        tracker = ExperimentTracker(config=cfg, backend="mlflow")
    else:
        if required:
            raise ValueError("No experiment tracking backend configured")
        else:
            print("No experiment tracking configured, using dummy tracker")
            tracker = None

    print("Experiment tracker initialized.")
    return tracker


def load_dataset(
    cfg: DictConfig,
    model,
    dataset_type: str = "train",
    limit: Optional[int] = None,
):
    """Load a dataset (train, validation, or test)."""
    from src.data.dataset import MixingDataset

    print(f"Loading {dataset_type} dataset...")

    # Determine paths based on dataset type
    if dataset_type == "train":
        jsonl_path = cfg.data.train_jsonl_path
        audio_root = cfg.data.train_audio_root
    elif dataset_type == "test":
        jsonl_path = cfg.data.test_jsonl_path
        audio_root = cfg.data.test_audio_root
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Use provided limit or config limit
    if limit is None:
        limit = cfg.data.get("limit")

    dataset = MixingDataset(
        jsonl_path=jsonl_path,
        audio_root=audio_root,
        tokenizer=model.tokenizer,
        sample_rate=cfg.data.audio.sample_rate,
        limit=limit,
        use_instruction=cfg.data.use_instruction,
    )

    print(f"{dataset_type.title()} dataset size: {len(dataset)} samples")

    # Quick validation
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample audio shape: {sample['audio'].shape}")

    print(f"{dataset_type.title()} dataset loaded.")
    return dataset
