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

        # Stack audio tensors (pad to same length if needed)
        if "audio" not in features[0]:
            raise KeyError("Expected 'audio' key in features[0]")
        audio_list = [feature["audio"] for feature in features]

        # Pad audio tensors to the same length
        batch["audio"] = self._pad_audio_tensors(audio_list)

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

    def _pad_audio_tensors(self, audio_list: List[torch.Tensor]) -> torch.Tensor:
        """Pad audio tensors to the same length."""
        if not audio_list:
            return torch.empty(0)

        # Find the maximum length
        max_length = max(audio.shape[-1] for audio in audio_list)

        # Pad or crop each tensor to max_length
        padded_audio = []
        for audio in audio_list:
            if audio.shape[-1] < max_length:
                # Pad with zeros
                padding_size = max_length - audio.shape[-1]
                if len(audio.shape) == 1:
                    # 1D tensor: pad at the end
                    padded = torch.nn.functional.pad(
                        audio, (0, padding_size), mode="constant", value=0
                    )
                else:
                    # Multi-dimensional: pad the last dimension
                    pad_width = [0, 0] * (len(audio.shape) - 1) + [0, padding_size]
                    padded = torch.nn.functional.pad(
                        audio, pad_width, mode="constant", value=0
                    )
                padded_audio.append(padded)
            elif audio.shape[-1] > max_length:
                # Crop to max_length
                if len(audio.shape) == 1:
                    # 1D tensor: crop the last dimension
                    cropped = audio[:max_length]
                else:
                    # Multi-dimensional: crop the last dimension
                    cropped = audio[..., :max_length]
                padded_audio.append(cropped)
            else:
                # Already the right length
                padded_audio.append(audio)

        # Stack the padded tensors
        return torch.stack(padded_audio, dim=0)
