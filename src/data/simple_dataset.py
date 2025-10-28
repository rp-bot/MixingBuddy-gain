"""
Simple dataset for instrument classification data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import librosa
from torch.utils.data import Dataset


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class SimpleInstrumentDataset(Dataset):
    """Simple dataset for instrument classification training.

    Each sample contains:
    - path: Path to audio file
    - instruction: Text instruction for the model
    - response: Expected response from the model (instrument name)
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        sample_rate: int = 32000,
        limit: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio
            limit: Optional limit on number of samples to load
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate

        # Load data
        self.data = load_jsonl(self.jsonl_path)
        if limit is not None:
            self.data = self.data[:limit]

        print(f"Loaded {len(self.data)} samples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[Dict]]]:
        """Get a single training sample."""
        item = self.data[idx]

        # Load audio at target sample rate
        # The path in JSONL already includes "data/" prefix, so we need to construct it correctly
        audio_path = self.audio_root.parent / item["path"]
        audio = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)[0]

        # Get instruction and response
        instruction = item["instruction"]
        response = item["response"]

        # Create messages in conversational format
        messages = []
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})

        return {
            "audio": torch.from_numpy(audio).float(),
            "messages": messages,
            "sample_rate": self.sample_rate,
            "instruction": instruction,
            "response": response,
        }
