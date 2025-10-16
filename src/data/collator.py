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
    This collator is designed to work with SFTTrainer and expects the dataset
    to return a 'messages' field in conversational format and an 'audio' field.

    This collator:
    - Applies the chat template to the 'messages' field.
    - Tokenizes the formatted text.
    - Creates labels for language modeling, masking the prompt part.
    - Pads text sequences (input_ids, attention_mask, labels) to the same length.
    - Pads and stacks audio tensors.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract messages and audio from features
        all_messages = [f["messages"] for f in features]
        audio_list = [f["audio"] for f in features]

        # Apply chat template and tokenize
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in all_messages
        ]
        tokenized_batch = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Create labels, setting prompt tokens to ignore index
        labels = tokenized_batch["input_ids"].clone()
        for i, messages in enumerate(all_messages):
            # To properly calculate the loss, we need to determine where the
            # prompt ends and the response begins. We do this by applying the chat
            # template to the prompt part only (all messages except the last one).
            user_part = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            user_tokens_len = len(
                self.tokenizer(user_part, add_special_tokens=False)["input_ids"]
            )

            # Set the ignore index for tokens up to the start of the assistant's response
            labels[i, :user_tokens_len] = -100

        tokenized_batch["labels"] = labels

        # Pad audio tensors to the same length
        tokenized_batch["audio"] = self._pad_audio_tensors(audio_list)

        return tokenized_batch

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
