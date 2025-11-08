import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.encoders.encodec import EncodecEncoder
from src.models.encoders.mert import MERTEncoder
from src.models.projections import MLPProjection


logger = logging.getLogger(__name__)


class ModularMultimodalModel(nn.Module):
    """Multimodal model for fine-tuning and inference."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        use_qlora: bool = False,
        lora_config: Optional[Any] = None,
        llm: Any = None,
        tokenizer: Any = None,
        encoder_config: Optional[Dict[str, Any]] = None,
        projection_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_name = model_name

        self.llm = llm
        self.tokenizer = tokenizer
        self.config = llm.config

        if encoder_config is None:
            encoder_config = {
                "model_name": "facebook/encodec_24khz",
                "freeze": True,
                "device": None,
            }

        if encoder_config.get("device") is None:
            encoder_config["device"] = self.llm.device

        # Detect encoder type based on model_name
        encoder_model_name = encoder_config.get("model_name", "")
        if "MERT" in encoder_model_name or "m-a-p/MERT" in encoder_model_name:
            logger.info(f"Using MERT encoder: {encoder_model_name}")
            # Add input_sample_rate for MERT (default 32kHz from data config)
            if "input_sample_rate" not in encoder_config:
                encoder_config["input_sample_rate"] = 32000
            self.audio_encoder = MERTEncoder(**encoder_config)
        else:
            logger.info(f"Using EnCodec encoder: {encoder_model_name}")
            self.audio_encoder = EncodecEncoder(**encoder_config)

        llm_hidden_size = self.llm.config.hidden_size
        audio_hidden_size = self.audio_encoder.output_dim

        self.use_auxiliary_loss = False
        self.auxiliary_loss_weight = 0.01
        if projection_config is not None:
            self.use_auxiliary_loss = projection_config.get("use_auxiliary_loss", False)
            self.auxiliary_loss_weight = projection_config.get(
                "auxiliary_loss_weight", 0.01
            )

        if (
            projection_config is None
            or projection_config.get("type", "linear") == "linear"
        ):
            self.audio_projection = nn.Linear(audio_hidden_size, llm_hidden_size)
        elif projection_config.get("type") == "mlp":
            mlp_config = dict(projection_config)
            mlp_config.pop("type", None)
            mlp_config.pop("use_auxiliary_loss", None)
            mlp_config.pop("auxiliary_loss_weight", None)
            mlp_config.pop("freeze_projection", None)  # Not a parameter for MLPProjection
            self.audio_projection = MLPProjection(
                input_dim=audio_hidden_size, output_dim=llm_hidden_size, **mlp_config
            )
        else:
            raise ValueError(
                f"Unsupported projection type: {projection_config.get('type')}"
            )

        llm_device = self.llm.device
        llm_dtype = next(self.llm.get_input_embeddings().parameters()).dtype

        self.audio_encoder.to(llm_device)
        self.audio_projection.to(device=llm_device, dtype=llm_dtype)

        if self.use_auxiliary_loss:
            logger.info(
                "Auxiliary loss enabled with weight %.4f", self.auxiliary_loss_weight
            )

    def print_trainable_parameters(self) -> None:
        """Log parameter breakdown by component."""
        trainable_params = 0
        all_param = 0
        frozen_params = 0

        llm_params = 0
        llm_trainable = 0
        audio_encoder_params = 0
        audio_encoder_trainable = 0
        audio_projection_params = 0
        audio_projection_trainable = 0

        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

            if name.startswith("llm."):
                llm_params += param.numel()
                if param.requires_grad:
                    llm_trainable += param.numel()
            elif name.startswith("audio_encoder."):
                audio_encoder_params += param.numel()
                if param.requires_grad:
                    audio_encoder_trainable += param.numel()
            elif name.startswith("audio_projection."):
                audio_projection_params += param.numel()
                if param.requires_grad:
                    audio_projection_trainable += param.numel()

        # Format numbers with commas for readability
        def format_num(num):
            return f"{num:,}"

        logger.info("=" * 60)
        logger.info("MODEL PARAMETER SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"LLM:           {format_num(llm_params):>12} total, {format_num(llm_trainable):>12} trainable"
        )
        logger.info(
            f"Audio Encoder: {format_num(audio_encoder_params):>12} total, {format_num(audio_encoder_trainable):>12} trainable"
        )
        logger.info(
            f"Audio Proj:    {format_num(audio_projection_params):>12} total, {format_num(audio_projection_trainable):>12} trainable"
        )
        logger.info("-" * 60)
        logger.info(
            f"TOTAL:         {format_num(all_param):>12} total, {format_num(trainable_params):>12} trainable, {format_num(frozen_params):>12} frozen"
        )
        logger.info("=" * 60)

        # Memory usage estimate
        trainable_mb = trainable_params * 4 / (1024 * 1024)  # 4 bytes per float32
        total_mb = all_param * 4 / (1024 * 1024)
        logger.info(
            f"Memory estimate: {total_mb:.1f} MB total, {trainable_mb:.1f} MB trainable"
        )
        logger.info("=" * 60)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio with EnCodec; supports batched and unbatched input."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]
        encoded_list = []
        for i in range(batch_size):
            single_audio = audio[i]
            encoded = self.audio_encoder(single_audio)
            if encoded.dim() == 3:
                encoded = encoded.squeeze(0)
            encoded_list.append(encoded)
        return torch.stack(encoded_list, dim=0)

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        audio = kwargs.get("audio")
        labels = kwargs.get("labels")

        audio_features = self.encode_audio(audio)
        audio_features = audio_features.to(
            dtype=next(self.audio_projection.parameters()).dtype
        )
        projected_audio_embeds = self.audio_projection(audio_features)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        auxiliary_loss = torch.tensor(0.0, device=projected_audio_embeds.device)
        if self.use_auxiliary_loss and self.training:
            audio_norm_per_dim = (
                torch.norm(projected_audio_embeds, p=2, dim=-1)
                / projected_audio_embeds.shape[-1]
            )
            text_norm_per_dim = (
                torch.norm(text_embeds, p=2, dim=-1) / text_embeds.shape[-1]
            )
            auxiliary_loss = torch.nn.functional.mse_loss(
                audio_norm_per_dim.mean(), text_norm_per_dim.mean().detach()
            )

        num_audio_tokens = projected_audio_embeds.shape[1]
        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)
        text_only_embeds = text_embeds[:, num_audio_tokens:]
        inputs_embeds = torch.cat([projected_audio_embeds, text_only_embeds], dim=1)

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )

        if self.use_auxiliary_loss and self.training:
            weighted_aux_loss = self.auxiliary_loss_weight * auxiliary_loss
            output.loss = output.loss + weighted_aux_loss
            if hasattr(output, "auxiliary_loss"):
                output.auxiliary_loss = auxiliary_loss.item()

        return output

    @torch.no_grad()
    def generate(
        self,
        text_input: str,
        audio: torch.Tensor,
        max_new_tokens: int,
        system_message: str,
    ) -> str:
        device = self.llm.device

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text_input},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        audio = audio.to(device)

        audio_features = self.encode_audio(audio)
        audio_features = audio_features.to(
            dtype=next(self.audio_projection.parameters()).dtype
        )
        projected_audio_embeds = self.audio_projection(audio_features)

        text_embeds = self.llm.get_input_embeddings()(model_inputs.input_ids)
        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)

        inputs_embeds = torch.cat((projected_audio_embeds, text_embeds), dim=1)

        audio_attention_mask = torch.ones(
            projected_audio_embeds.shape[:2], dtype=torch.long, device=device
        )
        attention_mask = torch.cat(
            (audio_attention_mask, model_inputs.attention_mask), dim=1
        )

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
