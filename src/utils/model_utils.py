"""
Shared utilities for model initialization and configuration.
"""

import os
import warnings
import logging
from typing import Optional
from pathlib import Path
from omegaconf import DictConfig
from dotenv import load_dotenv

from transformers import AutoTokenizer

# Load environment variables from .env file if it exists
# Look for .env in project root (2 levels up from src/utils/)
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


def initialize_tokenizer(model_name: str, token: Optional[str] = None):
    """Initializes and configures the tokenizer."""
    # Use provided token, or check environment variable, or use None (will use cached login)
    hf_token = token or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
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

    # Get Hugging Face token from config or environment
    hf_token = cfg.model.get("hf_token") or os.getenv("HF_TOKEN")

    # Load model without quantization
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype="auto",
        token=hf_token,
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

    # Get Hugging Face token from config or environment
    hf_token = cfg.model.get("hf_token") or os.getenv("HF_TOKEN")

    # Load model with quantization
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype="auto",
        quantization_config=quantization_config,
        token=hf_token,
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


def generate_run_name(cfg: DictConfig):
    """Generate run name using naming convention without requiring a tracker."""
    # Check if custom name is provided
    custom_name = cfg.experiment_tracking.get("name")
    if custom_name:
        return custom_name

    # Extract components for naming convention
    # Check if model config has a specific name for experiment naming
    model_config_name = getattr(cfg.model, "config_name", None)
    if model_config_name:
        model_abbr = cfg.experiment_naming.naming.components.model_abbr.get(
            model_config_name, model_config_name.replace("_", "-")
        )
    else:
        # Fallback to original logic
        model_name = cfg.model.model_name
        model_abbr = cfg.experiment_naming.naming.components.model_abbr.get(
            model_name, model_name.lower().replace("-instruct", "").replace("-", "")
        )

    # LoRA configuration
    lora_config = cfg.model.lora
    rank = lora_config.r
    alpha = lora_config.lora_alpha
    lora_str = f"r{rank}a{alpha}"

    # Dataset identifier from config mapping
    dataset_path = cfg.data.train_jsonl_path
    if "musdb" in dataset_path.lower():
        dataset_abbr = "musdb"
    else:
        dataset_abbr = "custom"

    # Experiment type from config mapping
    if cfg.model.use_qlora:
        exp_type = cfg.experiment_naming.naming.components.exp_type.qlora
    else:
        exp_type = cfg.experiment_naming.naming.components.exp_type.lora

    # Construct run name: {exp_type}-{model_abbr}-{lora_config}-{dataset_abbr}
    run_name = f"{exp_type}-{model_abbr}-{lora_str}-{dataset_abbr}"

    return run_name


def initialize_experiment_tracker(cfg: DictConfig, required: bool = False):
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


class IterableDatasetWrapper:
    """Wrapper to make PyTorch Dataset compatible with HuggingFace Trainer.

    This wrapper prevents audio truncation by keeping data in PyTorch format
    and avoiding HuggingFace Dataset's default 1,024 sequence length limit.
    The audio data is only loaded when accessed via __getitem__, not materialized
    in memory all at once.
    """

    def __init__(self, pytorch_dataset):
        self.pytorch_dataset = pytorch_dataset
        # Store the underlying data for train_test_split compatibility
        self.data = pytorch_dataset

        # Add HuggingFace Dataset attributes for compatibility with SFTTrainer
        # Get column names from a sample item
        if len(pytorch_dataset) > 0:
            sample = pytorch_dataset[0]
            self.column_names = list(sample.keys())
        else:
            self.column_names = []

    def __len__(self):
        return len(self.pytorch_dataset)

    def __getitem__(self, idx):
        return self.pytorch_dataset[idx]

    def map(self, function, **kwargs):
        """
        Apply a function to all elements in the dataset.
        For compatibility with SFTTrainer, we return self since our collator
        handles all the necessary transformations.
        """
        # SFTTrainer uses map to apply tokenization, but we handle this in the collator
        # So we just return self unchanged
        return self

    def select(self, indices):
        """Select a subset of the dataset by indices."""
        from torch.utils.data import Subset

        subset = Subset(self.pytorch_dataset, indices)
        return IterableDatasetWrapper(subset)

    def filter(self, function, **kwargs):
        """Filter dataset based on a condition."""
        # For now, return self - add filtering logic if needed
        return self

    def train_test_split(self, test_size, seed):
        """Split dataset into train and test sets."""
        from torch.utils.data import Subset
        import random

        # Set seed for reproducibility
        random.seed(seed)

        # Create indices and shuffle
        indices = list(range(len(self.pytorch_dataset)))
        random.shuffle(indices)

        # Split indices
        split_idx = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        # Create subset datasets wrapped in our wrapper
        train_subset = Subset(self.pytorch_dataset, train_indices)
        test_subset = Subset(self.pytorch_dataset, test_indices)

        return {
            "train": IterableDatasetWrapper(train_subset),
            "test": IterableDatasetWrapper(test_subset),
        }


def load_dataset(
    cfg: DictConfig,
    dataset_type: str = "train",
    limit: Optional[int] = None,
    random_seed: Optional[int] = None,
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
        sample_rate=cfg.data.audio.sample_rate,
        limit=limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
        random_seed=random_seed,
    )

    print(f"{dataset_type.title()} dataset size: {len(dataset)} samples")

    # Quick validation
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample audio shape: {sample['audio'].shape}")

    print(f"{dataset_type.title()} dataset loaded.")
    return dataset
