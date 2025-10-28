import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

from src.utils.audio_utils import db_to_linear


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class MusdbDataset(Dataset):
    """Dataset for loading mixing training samples with on-the-fly flawed mix generation.

    Each sample contains:
    - flawed_mix: Synthesized mix with intentional errors (10 seconds) - generated on-the-fly
    - instruction: Text instruction for the model
    - response: Expected response from the model
    - reference_mix_path: Path to the reference mix for potential future use

    This dataset generates flawed mixes dynamically during __getitem__ instead of
    loading pre-generated audio files, saving disk space and enabling flexible
    experimentation with error injection parameters.
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
            audio_root: Root directory for audio files (unused in on-the-fly mode)
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
        """Get a single training sample with on-the-fly flawed mix generation."""
        item = self.data[idx]

        # Load individual stems for the chunk
        stems = {}
        for stem_name, stem_path in item["meta"]["paths"]["stems"].items():
            audio, _ = librosa.load(str(stem_path), sr=self.sample_rate, mono=True)
            # Extract chunk based on time reference with bounds checking
            start_sample = int(item["meta"]["time_ref"]["start_sec"] * self.sample_rate)
            end_sample = int(item["meta"]["time_ref"]["end_sec"] * self.sample_rate)

            # Ensure indices are within bounds
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(start_sample, min(end_sample, len(audio)))

            # Extract chunk
            chunk = audio[start_sample:end_sample]

            # If chunk is shorter than expected, pad with zeros
            expected_length = int(
                (
                    item["meta"]["time_ref"]["end_sec"]
                    - item["meta"]["time_ref"]["start_sec"]
                )
                * self.sample_rate
            )
            if len(chunk) < expected_length:
                padding_length = expected_length - len(chunk)
                chunk = np.pad(
                    chunk, (0, padding_length), mode="constant", constant_values=0
                )

            stems[stem_name] = chunk

        # Apply error to target stem
        target_stem = item["meta"]["target_stem"]
        error_category = item["meta"]["error_category"]
        intended_gain_db = item["meta"]["intended_gain_db"]

        if error_category != "no_error" and intended_gain_db != 0.0:
            # Apply gain error using intended_gain_db from metadata
            gain_linear = db_to_linear(intended_gain_db)
            stems[target_stem] = stems[target_stem] * gain_linear

        # Create flawed mix by summing stems
        flawed_mix = sum(stems.values())

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
