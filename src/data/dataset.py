import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import librosa
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class MixingDataset(Dataset):
    """Dataset for loading mixing training samples.

    Each sample contains:
    - flawed_mix: Synthesized mix with intentional errors (10 seconds)
    - instruction: Text instruction for the model
    - response: Expected response from the model
    - reference_mix_path: Path to the reference mix for potential future use
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        sample_rate: int,
        max_length: int = 512,
        limit: Optional[int] = None,
        use_instruction: bool = True,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            tokenizer: HuggingFace tokenizer
            sample_rate: Target sample rate for audio (required)
            max_length: Maximum sequence length for tokenization
            limit: Optional limit on number of samples to load
            use_instruction: Whether to include instruction text in training
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.use_instruction = use_instruction

        # Load data
        self.data = load_jsonl(self.jsonl_path)
        if limit is not None:
            self.data = self.data[:limit]

        print(f"Loaded {len(self.data)} samples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        item = self.data[idx]

        # Load flawed mix at target sample rate
        flawed_mix_path = item["flawed_mix_path"]
        flawed_mix = librosa.load(str(flawed_mix_path), sr=self.sample_rate, mono=True)[
            0
        ]

        # Get instruction and response
        instruction = item["instruction"]
        response = item["response"]

        # Tokenize based on whether we use instructions
        if self.use_instruction:
            # Current behavior: instruction + separator + response
            full_text = instruction + self.tokenizer.eos_token + response
            # Calculate instruction length for masking
            instruction_tokenized = self.tokenizer(
                instruction + self.tokenizer.eos_token,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            instruction_len = instruction_tokenized["input_ids"].shape[1]
        else:
            # New behavior: separator + response (maintains separator pattern)
            full_text = self.tokenizer.eos_token + response
            # Calculate eos_token length for masking (same as instruction case)
            eos_tokenized = self.tokenizer(
                self.tokenizer.eos_token,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            instruction_len = eos_tokenized["input_ids"].shape[1]  # Mask the eos_token

        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Create labels, ignoring the instruction part (if any)
        labels = input_ids.clone()
        labels[:instruction_len] = -100  # -100 is the ignore index for CrossEntropyLoss

        return {
            "audio": torch.from_numpy(flawed_mix).float(),
            "sample_rate": self.sample_rate,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instruction": instruction,
            "response": response,
            "reference_mix_path": item["reference_mix_path"],
            "target_stem": item["meta"]["target_stem"],
            "error_category": item["meta"]["error_category"],
            "global_uid": item["global_uid"],
        }

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio (for debugging)."""
        return self.data[idx]
