"""
DPO Data collator for multimodal (text + audio) preference optimization.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DPOMultimodalDataCollator:
    """
    Data collator for DPO training with multimodal inputs.
    
    Handles both chosen and rejected responses for the same audio input.
    Creates separate tokenized batches for chosen and rejected, but shares
    the same audio tensor since the input is identical.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    audio_encoder_stride: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of DPO samples.
        
        Each feature contains:
        - audio: Audio tensor
        - chosen_messages: Messages list for chosen response
        - rejected_messages: Messages list for rejected response
        - global_uid: (optional) Sample identifier
        
        Returns:
        - audio: Padded audio tensor (shared for both chosen and rejected)
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
        - global_uid: (optional) Sample identifiers
        """
        # Extract data
        audio_list = [f["audio"] for f in features]
        chosen_messages = [f["chosen_messages"] for f in features]
        rejected_messages = [f["rejected_messages"] for f in features]

        # Pad audio tensors (shared for both chosen and rejected)
        padded_audio = self._pad_audio_tensors(audio_list)
        
        # Determine the number of audio tokens to prepend
        num_audio_tokens = 0
        if self.audio_encoder_stride is not None and padded_audio.nelement() > 0:
            max_samples = padded_audio.shape[-1]
            num_audio_tokens = math.ceil(max_samples / self.audio_encoder_stride)

        # Process chosen responses
        chosen_batch = self._process_messages(
            chosen_messages, num_audio_tokens, prefix="chosen"
        )

        # Process rejected responses
        rejected_batch = self._process_messages(
            rejected_messages, num_audio_tokens, prefix="rejected"
        )

        # Combine into final batch
        batch = {
            "audio": padded_audio,
            **chosen_batch,
            **rejected_batch,
        }
        
        # Pass through global_uid if available (for logging)
        if "global_uid" in features[0]:
            batch["global_uid"] = [f["global_uid"] for f in features]

        return batch

    def _process_messages(
        self, messages_list: List[List[Dict]], num_audio_tokens: int, prefix: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process messages (either chosen or rejected) into tokenized format.
        
        Args:
            messages_list: List of message lists (one per sample)
            num_audio_tokens: Number of dummy tokens to prepend for audio
            prefix: Prefix for keys ("chosen" or "rejected")
            
        Returns:
            Dictionary with {prefix}_input_ids, {prefix}_attention_mask, {prefix}_labels
        """
        # Apply chat template and tokenize
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in messages_list
        ]
        
        text_tokenized = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Create labels, masking prompt tokens
        labels = text_tokenized["input_ids"].clone()
        for i, messages in enumerate(messages_list):
            # Get prompt part (all messages except the last assistant response)
            user_part = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            user_tokens_len = len(
                self.tokenizer(user_part, add_special_tokens=False)["input_ids"]
            )
            # Mask prompt tokens
            labels[i, :user_tokens_len] = -100

        # Prepend dummy tokens for audio features
        if num_audio_tokens > 0:
            batch_size = text_tokenized["input_ids"].shape[0]
            
            dummy_input_ids = torch.full(
                (batch_size, num_audio_tokens),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            dummy_attention_mask = torch.ones(
                (batch_size, num_audio_tokens), dtype=torch.long
            )
            dummy_labels = torch.full(
                (batch_size, num_audio_tokens), -100, dtype=torch.long
            )

            input_ids = torch.cat([dummy_input_ids, text_tokenized["input_ids"]], dim=1)
            attention_mask = torch.cat(
                [dummy_attention_mask, text_tokenized["attention_mask"]], dim=1
            )
            labels = torch.cat([dummy_labels, labels], dim=1)
        else:
            input_ids = text_tokenized["input_ids"]
            attention_mask = text_tokenized["attention_mask"]

        return {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attention_mask": attention_mask,
            f"{prefix}_labels": labels,
        }

    def _pad_audio_tensors(self, audio_list: List[Any]) -> torch.Tensor:
        """Pad audio tensors to the same length."""
        if not audio_list:
            return torch.empty(0)

        # Convert to tensors if needed
        audio_list = [
            audio.clone().float()
            if isinstance(audio, torch.Tensor)
            else torch.tensor(audio, dtype=torch.float32)
            for audio in audio_list
        ]

        # Find the maximum length
        max_length = max(audio.shape[-1] for audio in audio_list)

        # Pad or crop each tensor to max_length
        padded_audio = []
        for audio in audio_list:
            if audio.shape[-1] < max_length:
                # Pad with zeros
                padding_size = max_length - audio.shape[-1]
                if len(audio.shape) == 1:
                    padded = torch.nn.functional.pad(
                        audio, (0, padding_size), mode="constant", value=0
                    )
                else:
                    pad_width = [0, 0] * (len(audio.shape) - 1) + [0, padding_size]
                    padded = torch.nn.functional.pad(
                        audio, pad_width, mode="constant", value=0
                    )
                padded_audio.append(padded)
            elif audio.shape[-1] > max_length:
                # Crop to max_length
                if len(audio.shape) == 1:
                    cropped = audio[:max_length]
                else:
                    cropped = audio[..., :max_length]
                padded_audio.append(cropped)
            else:
                padded_audio.append(audio)

        # Stack the padded tensors
        return torch.stack(padded_audio, dim=0)

