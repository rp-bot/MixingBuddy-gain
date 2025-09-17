"""
Dataset classes and data processing utilities for LLM fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class AutomaticMixingDataset(Dataset):
    """Dataset class for automatic mixing data."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of data samples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add special tokens
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        sample = self.data[idx]

        # Format the text for causal language modeling
        text = self._format_text(sample)

        # Debug: Log the text being tokenized
        logger.debug(f"Sample {idx} text length: {len(text)} characters")
        logger.debug(f"Sample {idx} text preview: {text[:200]}...")

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        # Debug: Log tokenization results
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        logger.debug(f"Sample {idx} tokenized length: {len(input_ids)} tokens")
        logger.debug(f"Sample {idx} input_ids shape: {input_ids.shape}")

        # For causal LM, labels start as input_ids, then mask padding positions
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_text(self, sample: Dict[str, Any]) -> str:
        """Format a single sample into text for training."""
        # This is a template - customize based on your data format
        if "instruction" in sample and "response" in sample:
            # Instruction-following format
            return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
        elif "input" in sample and "output" in sample:
            # Input-output format
            return f"Input: {sample['input']}\nOutput: {sample['output']}"
        elif "text" in sample:
            # Plain text format
            return sample["text"]
        else:
            # Fallback - use all available fields
            return " ".join(
                [f"{k}: {v}" for k, v in sample.items() if isinstance(v, str)]
            )

    @classmethod
    def from_jsonl(
        cls, file_path: Union[str, Path], **kwargs
    ) -> "AutomaticMixingDataset":
        """Load dataset from JSONL file."""
        file_path = Path(file_path)

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return cls(data, **kwargs)

    @classmethod
    def from_json(
        cls, file_path: Union[str, Path], **kwargs
    ) -> "AutomaticMixingDataset":
        """Load dataset from JSON file."""
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            pass  # Already a list
        elif isinstance(data, dict):
            # Assume it's a dict with data in a key
            data = data.get("data", list(data.values())[0] if data else [])
        else:
            raise ValueError(f"Unexpected data format in {file_path}")

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return cls(data, **kwargs)


class DataProcessor:
    """Data processing utilities."""

    def __init__(self, config: DictConfig):
        """
        Initialize data processor.

        Args:
            config: Data configuration from Hydra
        """
        self.config = config

    def load_dataset(self, split: str = "train") -> AutomaticMixingDataset:
        """Load dataset for a specific split."""
        dataset_config = self.config.dataset

        if split == "train":
            data_path = Path(dataset_config.path) / dataset_config.train_file
        elif split == "validation":
            data_path = Path(dataset_config.path) / dataset_config.validation_file
        elif split == "test":
            data_path = Path(dataset_config.path) / dataset_config.test_file
        else:
            raise ValueError(f"Unknown split: {split}")

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer.name,
            padding_side=self.config.tokenizer.padding_side,
            truncation_side=self.config.tokenizer.truncation_side,
            use_fast=self.config.tokenizer.use_fast,
        )

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load dataset
        dataset = AutomaticMixingDataset.from_jsonl(
            data_path,
            tokenizer=tokenizer,
            max_length=self.config.processing.max_length,
            padding=self.config.processing.padding,
            truncation=self.config.processing.truncation,
            add_special_tokens=self.config.processing.add_special_tokens,
        )

        logger.info(f"Loaded {split} dataset with {len(dataset)} samples")
        return dataset

    def create_dataloader(
        self, dataset: AutomaticMixingDataset, split: str = "train"
    ) -> DataLoader:
        """Create DataLoader for a dataset."""
        dataloader_config = self.config.dataloader

        # Adjust batch size for validation/test
        batch_size = dataloader_config.batch_size
        if split in ["validation", "test"]:
            batch_size = min(batch_size, 8)  # Smaller batch size for eval

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=dataloader_config.shuffle and split == "train",
            num_workers=dataloader_config.num_workers,
            pin_memory=dataloader_config.pin_memory,
            drop_last=dataloader_config.drop_last and split == "train",
        )

    def get_dataset_stats(self, dataset: AutomaticMixingDataset) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {
            "num_samples": len(dataset),
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "total_tokens": 0,
        }

        lengths = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Count non-padding tokens by checking attention mask
            attention_mask = sample["attention_mask"]
            length = attention_mask.sum().item()  # Count actual tokens (not padding)
            lengths.append(length)

        if lengths:
            stats.update(
                {
                    "avg_length": sum(lengths) / len(lengths),
                    "max_length": max(lengths),
                    "min_length": min(lengths),
                    "total_tokens": sum(lengths),
                }
            )

        return stats


def create_sample_data(output_dir: Path, num_samples: int = 100):
    """Create sample data for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample automatic mixing data
    sample_data = []
    for i in range(num_samples):
        sample = {
            "instruction": f"Analyze the mixing parameters for track {i + 1}",
            "response": f"Track {i + 1} requires EQ adjustments at 2kHz, compression with ratio 3:1, and reverb with 1.2s decay time.",
        }
        sample_data.append(sample)

    # Save as JSONL
    train_data = sample_data[: int(0.8 * num_samples)]
    val_data = sample_data[int(0.8 * num_samples) : int(0.9 * num_samples)]
    test_data = sample_data[int(0.9 * num_samples) :]

    for split, data in [
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ]:
        file_path = output_dir / f"{split}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")
        logger.info(f"Created {len(data)} samples in {file_path}")


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    tokenizer: PreTrainedTokenizer = None,
    max_length: int = 2048,
    **kwargs,
) -> AutomaticMixingDataset:
    """Load dataset from Hugging Face Hub."""
    logger.info(f"Loading dataset {dataset_name} from Hugging Face")

    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    # Convert to list of dicts
    data = [sample for sample in dataset]

    if tokenizer is None:
        from transformers import AutoTokenizer

        # Use a more appropriate default tokenizer - Qwen2-Audio for audio tasks
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return AutomaticMixingDataset(
        data=data, tokenizer=tokenizer, max_length=max_length, **kwargs
    )
