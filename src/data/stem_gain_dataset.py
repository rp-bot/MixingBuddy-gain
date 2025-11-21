"""
Dataset for stem classification and gain regression.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class StemGainDataset(Dataset):
    """Dataset for multi-label stem classification.
    
    Each sample contains:
    - audio: Flawed mix audio (10 seconds)
    - multi_label: 15 binary labels (3 stems × 5 categories)
    
    Stems: vocals, drums, bass
    Categories: very_quiet, quiet, balanced, loud, very_loud
    Total: 15 classes (vocals_very_quiet, vocals_quiet, ..., bass_very_loud)
    """
    
    # Stem and category mappings
    STEMS = ["vocals", "drums", "bass"]
    CATEGORIES = ["very_quiet", "quiet", "balanced", "loud", "very_loud"]
    
    # Map error_category from data to category index
    ERROR_CATEGORY_TO_INDEX = {
        "very_quiet": 0,
        "quiet": 1,
        "balanced": 2,  # For no_error cases
        "loud": 3,
        "very_loud": 4,
    }
    
    # Legacy mapping for backward compatibility (deprecated)
    STEM_TO_INDEX = {"vocals": 0, "drums": 1, "bass": 2, "no_error": 3}
    INDEX_TO_STEM = {0: "vocals", 1: "drums", 2: "bass", 3: "no_error"}
    
    @classmethod
    def get_num_classes(cls) -> int:
        """Get total number of classes (3 stems × 5 categories = 15)."""
        return len(cls.STEMS) * len(cls.CATEGORIES)
    
    @classmethod
    def get_class_index(cls, stem: str, category: str) -> int:
        """Get the class index for a stem-category combination.
        
        Args:
            stem: One of "vocals", "drums", "bass"
            category: One of "very_quiet", "quiet", "balanced", "loud", "very_loud"
        
        Returns:
            Class index in range [0, 14]
        """
        stem_idx = cls.STEMS.index(stem)
        category_idx = cls.CATEGORIES.index(category)
        return stem_idx * len(cls.CATEGORIES) + category_idx
    
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        audio_root: Union[str, Path],
        sample_rate: int = 32000,
        limit: Optional[int] = None,
        random_seed: Optional[int] = None,
        augmentation_config: Optional[Dict] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with training samples
            audio_root: Root directory for audio files
            sample_rate: Target sample rate for audio
            limit: Optional limit on number of samples to load
            random_seed: Optional random seed for reproducible random sampling
            augmentation_config: Configuration for on-the-fly augmentation
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.augmentation_config = augmentation_config
        
        # Load data
        self.data = load_jsonl(self.jsonl_path)
        
        # Filter out samples where target_stem is "other" (shouldn't happen, but safety check)
        self.data = [
            item for item in self.data 
            if item["meta"]["target_stem"] in self.STEMS
        ]
        
        if limit is not None:
            if random_seed is not None:
                random.seed(random_seed)
                self.data = random.sample(self.data, min(limit, len(self.data)))
            else:
                self.data = self.data[:limit]
        
        print(f"Loaded {len(self.data)} samples from {self.jsonl_path}")
        print(f"Multi-label classification: {len(self.STEMS)} stems × {len(self.CATEGORIES)} categories = {self.get_num_classes()} classes")
        print(f"Error category distribution: {self._get_error_category_distribution()}")
        if self.augmentation_config and self.augmentation_config.get("enable_gain_augmentation"):
            print(f"Augmentation enabled: Gain range +/- {self.augmentation_config.get('gain_augment_range_db', 2.0)} dB")
    
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
        
        # Apply augmentation if enabled
        if self.augmentation_config and self.augmentation_config.get("enable_gain_augmentation"):
            range_db = self.augmentation_config.get("gain_augment_range_db", 2.0)
            # Generate random gain in dB
            random_gain_db = random.uniform(-range_db, range_db)
            
            # Apply gain to audio: gain_linear = 10^(db/20)
            gain_linear = 10 ** (random_gain_db / 20.0)
            audio = audio * gain_linear
        
        # Build multi-label classification:
        #  - 15 binary labels (3 stems × 5 categories)
        #  - If error_category == "no_error": all stems are "balanced"
        #  - Else: target_stem gets the error_category, other stems are "balanced"
        error_category = item["meta"]["error_category"]
        target_stem = item["meta"]["target_stem"]
        
        # Initialize all labels to 0
        multi_label = torch.zeros(self.get_num_classes(), dtype=torch.float32)
        
        if error_category == "no_error":
            # All stems are balanced
            for stem in self.STEMS:
                class_idx = self.get_class_index(stem, "balanced")
                multi_label[class_idx] = 1.0
        else:
            # Map error_category to category (no_error -> balanced already handled above)
            if error_category in self.ERROR_CATEGORY_TO_INDEX:
                category = self.CATEGORIES[self.ERROR_CATEGORY_TO_INDEX[error_category]]
            else:
                # Fallback: treat unknown categories as "balanced"
                category = "balanced"
            
            # Set the target stem's category
            target_class_idx = self.get_class_index(target_stem, category)
            multi_label[target_class_idx] = 1.0
            
            # All other stems are balanced
            for stem in self.STEMS:
                if stem != target_stem:
                    balanced_class_idx = self.get_class_index(stem, "balanced")
                    multi_label[balanced_class_idx] = 1.0
        
        return {
            "audio": torch.from_numpy(audio).float(),
            "multi_label": multi_label,
            "target_stem": target_stem,
            "error_category": error_category,
            "global_uid": item["global_uid"],
        }
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample without loading audio (for debugging)."""
        return self.data[idx]
