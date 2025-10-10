import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.utils.audio_utils import load_audio_chunk, to_mono


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
    - anchor: Reference audio stem (10 seconds)
    - flawed_mix: Synthesized mix with intentional errors (10 seconds)
    - instruction: Text instruction for the model
    - response: Expected response from the model
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        sample_rate: int = 48000,  # Default to 48kHz to match synthesis
        silence_duration: float = 0.5,  # 0.5 seconds of silence between anchor and mix
        max_length: int = 512,
        limit: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            tokenizer: HuggingFace tokenizer
            sample_rate: Target sample rate for audio
            silence_duration: Duration of silence between anchor and mix (seconds)
            max_length: Maximum sequence length for tokenization
            limit: Optional limit on number of samples to load
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.max_length = max_length

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

        # Load anchor audio (reference stem)
        anchor_stem = item["meta"]["anchor_stem"]
        anchor_path = item["meta"]["paths"]["stems"][anchor_stem]
        anchor_full_path = anchor_path

        # Load anchor chunk (10 seconds)
        start_sec = item["meta"]["time_ref"]["start_sec"]
        end_sec = item["meta"]["time_ref"]["end_sec"]
        anchor = load_audio_chunk(
            str(anchor_full_path), start_sec, end_sec, self.sample_rate
        )

        # Load flawed mix
        flawed_mix_path = item["flawed_mix_path"]
        flawed_mix, _ = sf.read(str(flawed_mix_path))
        flawed_mix = to_mono(flawed_mix)

        # Resample if needed
        if len(flawed_mix) != len(anchor):
            # Simple resampling - just truncate or pad to match anchor length
            if len(flawed_mix) > len(anchor):
                flawed_mix = flawed_mix[: len(anchor)]
            else:
                # Pad with zeros
                padding = np.zeros(len(anchor) - len(flawed_mix), dtype=np.float32)
                flawed_mix = np.concatenate([flawed_mix, padding])

        # Create silence metadata between anchor and mix (don't concatenate here)
        silence_samples = int(self.silence_duration * self.sample_rate)

        # Get instruction and response
        instruction = item["instruction"]
        response = item["response"]

        # Tokenize instruction and response together
        # This is the standard way for training a causal LM.
        # The model learns to predict the response given the instruction.
        full_text = instruction + self.tokenizer.eos_token + response
        tokenized = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Create labels, ignoring the instruction part
        # We need to find where the instruction ends and the response begins
        instruction_tokenized = self.tokenizer(
            instruction + self.tokenizer.eos_token,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        instruction_len = instruction_tokenized["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:instruction_len] = -100  # -100 is the ignore index for CrossEntropyLoss

        return {
            "anchor_audio": torch.from_numpy(anchor).float(),
            "audio": torch.from_numpy(flawed_mix).float(),
            "silence_samples": silence_samples,
            "sample_rate": self.sample_rate,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instruction": instruction,
            "response": response,
            "anchor_stem": anchor_stem,
            "target_stem": item["meta"]["target_stem"],
            "error_category": item["meta"]["error_category"],
            "global_uid": item["global_uid"],
        }

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio (for debugging)."""
        return self.data[idx]
