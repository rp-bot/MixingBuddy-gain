"""
Dataset classes and data processing utilities for LLM fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
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
        audio_segment_duration: float = 3.0,  # Load only 3 seconds of audio
        sample_rate: int = 16000,  # Downsample to 16kHz for efficiency
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
            audio_segment_duration: Duration of audio segments to load (seconds)
            sample_rate: Target sample rate for audio
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.audio_segment_duration = audio_segment_duration
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        sample = self.data[idx]

        # Format the text for causal language modeling
        text = self._format_text(sample)

        # Load audio segments if available (disabled for test run)
        # audio_tensors = self._load_audio_segments(sample)
        audio_tensors = {}

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

        # For causal LM, labels are the same as input_ids
        encoding["labels"] = encoding["input_ids"].clone()

        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["labels"].squeeze(),
        }

        # Add audio tensors in the format expected by Qwen2-Audio
        if audio_tensors:
            # Qwen2-Audio expects 'audio' and 'audio_attention_mask' parameters
            # For now, let's just pass the first audio tensor as 'audio'
            if "audio_1" in audio_tensors:
                result["audio"] = audio_tensors["audio_1"]
                # Create attention mask for audio (all 1s for now)
                audio_length = audio_tensors["audio_1"].shape[-1]
                result["audio_attention_mask"] = torch.ones(
                    1, audio_length, dtype=torch.long
                )

        return result

    def _format_text(self, sample: Dict[str, Any]) -> str:
        """Format a single sample into text for training."""
        # For multimodal audio data, create a simplified instruction-response format
        if "instruction" in sample and "response" in sample:
            # Extract the core instruction without audio paths and metadata
            instruction = sample["instruction"]
            response = sample["response"]

            # Clean up the instruction by removing audio file paths and excessive metadata
            if "<|audio_start|>" in instruction:
                # Extract just the core instruction
                instruction = "Analyze the mixing balance issue in these audio inputs and identify the problem and solution."

            # Simplify response to avoid very long text
            if len(response) > 200:
                # Extract key information
                if "problem_info" in sample and isinstance(
                    sample["problem_info"], dict
                ):
                    problem_info = sample["problem_info"]
                    stem = problem_info.get("stem_name", "unknown")
                    problem = problem_info.get("problem_type", "unknown")
                    solution = problem_info.get("solution", "unknown")
                    response = f"The {stem} has a {problem} issue. Solution: {solution}"

            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        elif "input" in sample and "output" in sample:
            # Input-output format
            return f"Input: {sample['input'][:200]}\nOutput: {sample['output'][:200]}"
        elif "text" in sample:
            # Plain text format
            return sample["text"][:400]  # Limit text length
        else:
            # Fallback - use only string fields, limited length
            text_parts = []
            for k, v in sample.items():
                if isinstance(v, str) and k not in [
                    "audio_1",
                    "audio_2",
                    "audio_3",
                    "audio_4",
                    "input",
                ]:
                    text_parts.append(f"{k}: {v[:100]}")
            return " ".join(text_parts[:3])  # Limit to first 3 fields

    def _load_audio_segments(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Load audio segments from file paths in sample."""
        audio_tensors = {}

        # Look for audio file paths in the sample
        audio_keys = ["audio_1", "audio_2", "audio_3", "audio_4"]

        for i, audio_key in enumerate(audio_keys):
            if audio_key in sample and sample[audio_key]:
                audio_path = Path(sample[audio_key])
                if audio_path.exists():
                    try:
                        # Load audio segment
                        audio_data = self._load_audio_file(audio_path)
                        if audio_data is not None:
                            audio_tensors[f"audio_{i + 1}"] = audio_data
                    except Exception as e:
                        logger.warning(f"Failed to load audio {audio_path}: {e}")

        return audio_tensors

    def _load_audio_file(self, audio_path: Path) -> Optional[torch.Tensor]:
        """Load a short segment from an audio file."""
        try:
            # Load only the first few seconds of audio
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=self.audio_segment_duration,
                offset=0.0,  # Start from the beginning
            )

            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)

            # Pad or truncate to fixed length
            target_length = int(self.sample_rate * self.audio_segment_duration)
            if len(audio_tensor) < target_length:
                # Pad with zeros
                padding = target_length - len(audio_tensor)
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            elif len(audio_tensor) > target_length:
                # Truncate
                audio_tensor = audio_tensor[:target_length]

            return audio_tensor.unsqueeze(0)  # Add channel dimension

        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            return None

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
            audio_segment_duration=self.config.processing.get(
                "audio_segment_duration", 3.0
            ),
            sample_rate=self.config.processing.get("sample_rate", 16000),
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
            length = sample["input_ids"].sum().item()  # Count non-padding tokens
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

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return AutomaticMixingDataset(
        data=data, tokenizer=tokenizer, max_length=max_length, **kwargs
    )
