"""
Data collator for multimodal (text + audio) training.
"""

import math
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
    audio_encoder_stride: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract messages and audio from features
        all_messages = [f["messages"] for f in features]
        audio_list = [f["audio"] for f in features]

        # Pad audio tensors to the same length before tokenizing text
        # This is necessary to determine the number of audio tokens to prepend.
        padded_audio = self._pad_audio_tensors(audio_list)
        tokenized_batch = {"audio": padded_audio}

        # Determine the number of audio tokens to prepend
        num_audio_tokens = 0
        if self.audio_encoder_stride is not None and padded_audio.nelement() > 0:
            max_samples = padded_audio.shape[-1]
            num_audio_tokens = math.ceil(max_samples / self.audio_encoder_stride)

        # Apply chat template and tokenize text
        formatted_texts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in all_messages
        ]
        text_tokenized = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Create labels, setting prompt tokens to ignore index
        labels = text_tokenized["input_ids"].clone()
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

        # Prepend dummy tokens for audio features to input_ids, attention_mask, and labels
        if num_audio_tokens > 0:
            batch_size = text_tokenized["input_ids"].shape[0]
            # Use pad_token_id for dummy input_ids. These will be replaced by embeddings.
            dummy_input_ids = torch.full(
                (batch_size, num_audio_tokens),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            # The model should attend to the audio features
            dummy_attention_mask = torch.ones(
                (batch_size, num_audio_tokens), dtype=torch.long
            )
            # The model should not compute loss on the audio part
            dummy_labels = torch.full(
                (batch_size, num_audio_tokens), -100, dtype=torch.long
            )

            tokenized_batch["input_ids"] = torch.cat(
                [dummy_input_ids, text_tokenized["input_ids"]], dim=1
            )
            tokenized_batch["attention_mask"] = torch.cat(
                [dummy_attention_mask, text_tokenized["attention_mask"]], dim=1
            )
            tokenized_batch["labels"] = torch.cat([dummy_labels, labels], dim=1)
        else:
            tokenized_batch["input_ids"] = text_tokenized["input_ids"]
            tokenized_batch["attention_mask"] = text_tokenized["attention_mask"]
            tokenized_batch["labels"] = labels

        return tokenized_batch

    def _pad_audio_tensors(self, audio_list: List[Any]) -> torch.Tensor:
        """Pad audio tensors to the same length."""
        if not audio_list:
            return torch.empty(0)

        # HF datasets can convert tensors to lists. We need to convert them back.
        audio_list = [torch.tensor(audio, dtype=torch.float32) for audio in audio_list]

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
