"""
Minimal multimodal model for Phase 5: combines audio encoder + projection + HF LLM.

This model:
- Encodes raw audio to Encodec-like features with an audio encoder (`encode(audio)`)
- Projects features to the LLM embedding dimension
- Concatenates audio embeddings (prefix) with token embeddings
- Computes causal LM loss while masking the audio prefix with -100 labels
"""

from typing import Optional, Dict, Union

import torch
import torch.nn as nn


class MultimodalMixingModel(nn.Module):
    """Combine an audio encoder, a projection layer, and a HuggingFace LLM."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        projection: nn.Module,
        llm: nn.Module,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.projection = projection
        self.llm = llm

    @property
    def device(self) -> torch.device:
        # Best effort to report a consistent device
        return next(self.parameters()).device

    def forward(
        self,
        audio: Union[torch.Tensor, Dict[str, torch.Tensor]],
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio: Tensor of shape (batch, samples)
            input_ids: Tensor of shape (batch, text_len)
            labels: Optional tensor of shape (batch, text_len)
            attention_mask: Optional mask (batch, text_len). If None, will be ones.

        Returns:
            Dict including loss (if labels provided) and the LLM outputs.
        """
        # Expect dict with 'anchor' and 'mix' keys
        anchor = audio["anchor"]
        mix = audio["mix"]
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        if mix.dim() == 1:
            mix = mix.unsqueeze(0)
        batch_size = anchor.shape[0]

        # Encode both segments
        anchor_features = self.audio_encoder.encode(anchor)  # type: ignore[attr-defined]
        mix_features = self.audio_encoder.encode(mix)  # type: ignore[attr-defined]


        target_dtype = self.projection.projection.weight.dtype
        anchor_features = anchor_features.to(target_dtype)
        mix_features = mix_features.to(target_dtype)
        audio_embeds = self.projection(anchor_features, mix_features)

        # 3) Get text embeddings -> (batch, text_len, hidden)
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # Ensure both embeddings have the same dtype as the LLM's embedding weights
        llm_embed_weight = self.llm.get_input_embeddings().weight
        target_embed_dtype = llm_embed_weight.dtype
        if audio_embeds.dtype != target_embed_dtype:
            audio_embeds = audio_embeds.to(target_embed_dtype)
        if input_embeds.dtype != target_embed_dtype:
            input_embeds = input_embeds.to(target_embed_dtype)

        # 4) Concat along time dimension -> (batch, time + text_len, hidden)
        inputs_embeds = torch.cat([audio_embeds, input_embeds], dim=1)

        # Build attention mask (audio prefix is all ones)
        audio_len = audio_embeds.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_ids.shape[1]),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        audio_attn = torch.ones(
            (batch_size, audio_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_attention_mask = torch.cat([audio_attn, attention_mask], dim=1)

        # 5) Prepare labels: mask audio prefix with -100 so it doesn't contribute to loss
        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": combined_attention_mask,
        }

        if labels is not None:
            ignore_prefix = torch.full(
                (batch_size, audio_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([ignore_prefix, labels], dim=1)
            model_kwargs["labels"] = combined_labels

        outputs = self.llm(**model_kwargs)

        # Standardize return to include loss if present
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return {"loss": outputs.loss, "logits": outputs.logits}
        return {"logits": outputs.logits}


def build_minimal_multimodal(
    audio_encoder: nn.Module,
    projection: nn.Module,
    llm: nn.Module,
) -> MultimodalMixingModel:
    """Helper to construct the minimal multimodal model."""
    return MultimodalMixingModel(
        audio_encoder=audio_encoder, projection=projection, llm=llm
    )
