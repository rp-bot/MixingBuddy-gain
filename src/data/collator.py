"""
Data collator for multimodal (text + audio) training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class MultimodalDataCollator:
    """
    Data collator that handles both text and audio inputs.

    This collator:
    - Pads text sequences (input_ids, attention_mask, labels) to the same length
    - Stacks audio tensors (assumes all audio has the same length)
    - Preserves other metadata fields
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.

        Args:
            features: List of samples from the dataset, each containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Labels for language modeling
                - audio: Audio tensor (all same length)
                - anchor_audio: Reference audio (optional)
                - Other metadata fields

        Returns:
            Batched dictionary with padded text and stacked audio
        """
        # Separate text and audio/metadata fields
        text_keys = ["input_ids", "attention_mask", "labels"]
        metadata_keys = [
            "silence_samples",
            "sample_rate",
            "instruction",
            "response",
            "anchor_stem",
            "target_stem",
            "error_category",
            "global_uid",
        ]

        # Extract text features for padding
        text_features = [
            {key: feature[key] for key in text_keys if key in feature}
            for feature in features
        ]

        # Pad text sequences using the tokenizer's pad method
        batch = self.tokenizer.pad(
            text_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Stack audio tensors (all audio should have the same length)
        if "audio" not in features[0]:
            raise KeyError("Expected 'audio' key in features[0]")
        audio_list = [feature["audio"] for feature in features]
        batch["audio"] = torch.stack(audio_list, dim=0)

        # Preserve metadata (take from first sample or collect all)
        for meta_key in metadata_keys:
            if meta_key in features[0]:
                # For scalar values, collect all
                if isinstance(features[0][meta_key], (int, float)):
                    batch[meta_key] = torch.tensor(
                        [feature[meta_key] for feature in features]
                    )
                # For strings, keep as list
                else:
                    batch[meta_key] = [feature[meta_key] for feature in features]

        return batch
