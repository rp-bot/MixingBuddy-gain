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
        sample_rate: int,
        system_message: str,
        use_instructions: bool,
        limit: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio (required)
            system_message: The system message to prepend to the conversation
            use_instructions: Whether to include instruction text in training
            limit: Optional limit on number of samples to load
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.use_instructions = use_instructions
        self.system_message = system_message

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

        # Load flawed mix at target sample rate
        flawed_mix_path = item["flawed_mix_path"]
        flawed_mix = librosa.load(str(flawed_mix_path), sr=self.sample_rate, mono=True)[
            0
        ]

        # Get instruction and response
        instruction = item["instruction"] if self.use_instructions else ""
        response = item["response"]

        # Create messages in conversational format
        messages = []
        messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})

        return {
            "audio": torch.from_numpy(flawed_mix).float(),
            "messages": messages,
            "sample_rate": self.sample_rate,
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
