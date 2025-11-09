"""
DPO Dataset loader for Direct Preference Optimization training.

Loads preference pairs (chosen vs rejected responses) for the same audio input.
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


class DPODataset(Dataset):
    """
    Dataset for loading DPO (Direct Preference Optimization) training pairs.

    Each sample contains:
    - audio: Audio tensor (flawed mix)
    - instruction: Text instruction for the model
    - chosen: Preferred/correct response
    - rejected: Non-preferred/incorrect response
    - meta: Additional metadata about the pair
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        sample_rate: int,
        system_message: str,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to DPO JSONL file with preference pairs
            audio_root: Root directory for audio files (typically not used since paths are absolute)
            sample_rate: Target sample rate for audio
            system_message: The system message to prepend to conversations
            limit: Optional limit on number of pairs to load
            random_seed: Optional random seed for reproducible random sampling
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.system_message = system_message

        # Load data
        self.data = load_jsonl(self.jsonl_path)
        
        if limit is not None:
            if random_seed is not None:
                import random
                random.seed(random_seed)
                self.data = random.sample(self.data, min(limit, len(self.data)))
            else:
                self.data = self.data[:limit]

        print(f"Loaded {len(self.data)} DPO pairs from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[Dict], str]]:
        """Get a single DPO training pair."""
        item = self.data[idx]

        # Load audio at target sample rate (same audio for both chosen and rejected)
        audio_path = item["audio_path"]
        audio = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)[0]

        # Get instruction and responses
        instruction = item["instruction"]
        chosen_response = item["chosen"]
        rejected_response = item["rejected"]

        # Create messages for chosen response
        chosen_messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": chosen_response},
        ]

        # Create messages for rejected response
        rejected_messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": rejected_response},
        ]

        return {
            "audio": torch.from_numpy(audio).float(),
            "sample_rate": self.sample_rate,
            "instruction": instruction,
            "chosen_messages": chosen_messages,
            "rejected_messages": rejected_messages,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "global_uid": item["global_uid"],
            "meta": item["meta"],
        }

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio (for debugging)."""
        return self.data[idx]


class DPODatasetWithSFT(Dataset):
    """
    Hybrid dataset that can be used for both SFT and DPO training.
    
    This is useful if you want to do staged training where you first do
    supervised fine-tuning on chosen responses, then switch to DPO.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        sample_rate: int,
        system_message: str,
        mode: str = "dpo",  # "sft" or "dpo"
        limit: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to DPO JSONL file
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio
            system_message: System message to prepend
            mode: "sft" (supervised fine-tuning on chosen only) or "dpo" (preference pairs)
            limit: Optional limit on number of samples
            random_seed: Optional random seed for reproducible sampling
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.system_message = system_message
        self.mode = mode

        # Load data
        self.data = load_jsonl(self.jsonl_path)
        
        if limit is not None:
            if random_seed is not None:
                import random
                random.seed(random_seed)
                self.data = random.sample(self.data, min(limit, len(self.data)))
            else:
                self.data = self.data[:limit]

        print(f"Loaded {len(self.data)} samples from {self.jsonl_path} (mode: {mode})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[Dict], str]]:
        """Get a sample (format depends on mode)."""
        item = self.data[idx]

        # Load audio
        audio_path = item["audio_path"]
        audio = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)[0]

        instruction = item["instruction"]

        if self.mode == "sft":
            # SFT mode: only return chosen response
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": item["chosen"]},
            ]
            return {
                "audio": torch.from_numpy(audio).float(),
                "messages": messages,
                "sample_rate": self.sample_rate,
                "instruction": instruction,
                "response": item["chosen"],
                "global_uid": item["global_uid"],
            }
        else:
            # DPO mode: return both chosen and rejected
            chosen_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": item["chosen"]},
            ]
            rejected_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": item["rejected"]},
            ]
            return {
                "audio": torch.from_numpy(audio).float(),
                "sample_rate": self.sample_rate,
                "instruction": instruction,
                "chosen_messages": chosen_messages,
                "rejected_messages": rejected_messages,
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "global_uid": item["global_uid"],
                "meta": item["meta"],
            }

