"""
Dataset for stem classification and gain regression.
"""

import json
import random
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


class StemGainDataset(Dataset):
    """Dataset for stem classification and gain regression.
    
    Each sample contains:
    - audio: Flawed mix audio (10 seconds)
    - target_stem: Which stem needs adjustment (vocals, drums, or bass)
    - intended_gain_db: Required gain adjustment in dB
    """
    
    # Classification label mapping
    #  - vocals/drums/bass: a specific stem needs adjustment
    #  - no_error: mix is balanced, no stem needs adjustment
    STEM_TO_INDEX = {"vocals": 0, "drums": 1, "bass": 2, "no_error": 3}
    INDEX_TO_STEM = {0: "vocals", 1: "drums", 2: "bass", 3: "no_error"}
    
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        sample_rate: int = 32000,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio
            limit: Optional limit on number of samples to load
            random_seed: Optional random seed for reproducible random sampling
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        
        # Load data
        self.data = load_jsonl(self.jsonl_path)
        
        # Filter out samples where target_stem is "other" (shouldn't happen, but safety check)
        self.data = [
            item for item in self.data 
            if item["meta"]["target_stem"] in self.STEM_TO_INDEX
        ]
        
        if limit is not None:
            if random_seed is not None:
                random.seed(random_seed)
                self.data = random.sample(self.data, min(limit, len(self.data)))
            else:
                self.data = self.data[:limit]
        
        print(f"Loaded {len(self.data)} samples from {self.jsonl_path}")
        print(f"Class distribution (vocals/drums/bass/no_error): {self._get_label_distribution()}")
        print(f"Error category distribution: {self._get_error_category_distribution()}")
    
    def _get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of classification labels (including no_error)."""
        distribution: Dict[str, int] = {}
        for item in self.data:
            error_category = item["meta"]["error_category"]
            if error_category == "no_error":
                label = "no_error"
            else:
                label = item["meta"]["target_stem"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def _get_error_category_distribution(self) -> Dict[str, int]:
        """Get distribution of error categories (no_error, quiet, very_quiet, loud, very_loud)."""
        distribution: Dict[str, int] = {}
        for item in self.data:
            error_cat = item["meta"]["error_category"]
            distribution[error_cat] = distribution.get(error_cat, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        """Get a single training sample."""
        item = self.data[idx]
        
        # Load flawed mix at target sample rate.
        # Note: flawed_mix_path in the JSONL is already a project-relative path
        # like "data/musdb18hq_processed/train/flawed_mixes/...", so we do NOT
        # prefix it with audio_root here to avoid double paths.
        flawed_mix_path = Path(item["flawed_mix_path"])
        audio = librosa.load(str(flawed_mix_path), sr=self.sample_rate, mono=True)[0]
        
        # Build classification label:
        #  - If error_category == "no_error": label = "no_error"
        #  - Else: label = target_stem (vocals/drums/bass)
        error_category = item["meta"]["error_category"]
        if error_category == "no_error":
            stem_index = self.STEM_TO_INDEX["no_error"]
            target_stem = "no_error"
        else:
            target_stem = item["meta"]["target_stem"]
            stem_index = self.STEM_TO_INDEX[target_stem]
        
        # Get intended gain in dB
        intended_gain_db = float(item["meta"]["intended_gain_db"])
        
        return {
            "audio": torch.from_numpy(audio).float(),
            "stem_label": stem_index,
            "gain_label": intended_gain_db,
            "target_stem": target_stem,
            "global_uid": item["global_uid"],
        }
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio (for debugging)."""
        return self.data[idx]

