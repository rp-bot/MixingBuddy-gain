import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.encoders.encodec import EncodecEncoder
from src.models.encoders.mert import MERTEncoder
from src.models.projections import MLPProjection
from src.models.projections.cross_attention import CrossAttentionProjection
from src.models.projections.qformer import QFormerProjection


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
        audio_augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_name = model_name

        self.llm = llm
        self.tokenizer = tokenizer
        self.config = llm.config

        # Add attributes needed for HuggingFace Trainer compatibility
        self._keys_to_ignore_on_save = []

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
        elif projection_config.get("type") == "cross_attention":
            cross_attn_config = dict(projection_config)
            cross_attn_config.pop("type", None)
            cross_attn_config.pop("use_auxiliary_loss", None)
            cross_attn_config.pop("auxiliary_loss_weight", None)
            cross_attn_config.pop("freeze_projection", None)  # Not a parameter for CrossAttentionProjection
            self.audio_projection = CrossAttentionProjection(
                feature_dim=audio_hidden_size, output_dim=llm_hidden_size, **cross_attn_config
            )
            self.use_cross_attention = True
        elif projection_config.get("type") == "qformer":
            qformer_config = dict(projection_config)
            qformer_config.pop("type", None)
            qformer_config.pop("use_auxiliary_loss", None)
            qformer_config.pop("auxiliary_loss_weight", None)
            freeze_projection = qformer_config.pop("freeze_projection", False)
            pretrained_path = qformer_config.pop("pretrained_path", None)
            mert_layer_weights_path = qformer_config.pop("mert_layer_weights_path", None)
            
            self.audio_projection = QFormerProjection(
                audio_dim=audio_hidden_size, 
                output_dim=llm_hidden_size, 
                **qformer_config
            )
            
            # Load pre-trained Q-Former weights if specified
            if pretrained_path:
                logger.info(f"Loading pre-trained Q-Former weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.audio_projection.load_state_dict(state_dict)
                logger.info("Pre-trained Q-Former weights loaded successfully")
            
            # Load pre-trained MERT layer weights if specified
            if mert_layer_weights_path and hasattr(self.audio_encoder, "layer_weights"):
                logger.info(f"Loading pre-trained MERT layer weights from {mert_layer_weights_path}")
                mert_state = torch.load(mert_layer_weights_path, map_location="cpu")
                # Support both formats:
                # 1. {"layer_weights": tensor} from alignment training
                # 2. Full encoder state dict from 06_train_model.py (mert_encoder.bin)
                if "layer_weights" in mert_state:
                    # Simple format from alignment training
                    self.audio_encoder.layer_weights.data = mert_state["layer_weights"].to(
                        self.audio_encoder.layer_weights.device
                    )
                elif isinstance(mert_state, dict) and any("layer_weights" in k for k in mert_state.keys()):
                    # Full encoder state dict - extract layer_weights
                    for key, value in mert_state.items():
                        if "layer_weights" in key:
                            self.audio_encoder.layer_weights.data = value.to(
                                self.audio_encoder.layer_weights.device
                            )
                            break
                else:
                    # Assume it's a full state dict that can be loaded directly
                    self.audio_encoder.load_state_dict(mert_state, strict=False)
                
                # Ensure layer weights stay frozen if freeze_layer_weights is set
                if self.audio_encoder.freeze_layer_weights:
                    self.audio_encoder.layer_weights.requires_grad = False
                    logger.info("MERT layer weights loaded and frozen (freeze_layer_weights=True)")
                else:
                    logger.info("Pre-trained MERT layer weights loaded (trainable)")
            
            # Freeze Q-Former if specified
            if freeze_projection:
                logger.info("Freezing Q-Former projection (pre-trained weights will not be updated)")
                for param in self.audio_projection.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(
                f"Unsupported projection type: {projection_config.get('type')}"
            )
        
        # Track if we're using cross-attention projection
        if not hasattr(self, "use_cross_attention"):
            self.use_cross_attention = False

        llm_device = self.llm.device
        llm_dtype = next(self.llm.get_input_embeddings().parameters()).dtype

        self.audio_encoder.to(llm_device)
        self.audio_projection.to(device=llm_device, dtype=llm_dtype)

        if self.use_auxiliary_loss:
            logger.info(
                "Auxiliary loss enabled with weight %.4f", self.auxiliary_loss_weight
            )

        # Audio augmentation configuration
        self.audio_augmentation_config = audio_augmentation_config
        if audio_augmentation_config and audio_augmentation_config.get("enabled", False):
            gain_range_db = audio_augmentation_config.get("gain_range_db", 3.0)
            noise_snr_db_range = audio_augmentation_config.get("noise_snr_db_range", None)
            dc_offset_range = audio_augmentation_config.get("dc_offset_range", None)
            
            aug_info = [f"gain: ±{gain_range_db:.1f} dB"]
            if noise_snr_db_range:
                aug_info.append(f"noise: {noise_snr_db_range[0]:.1f}-{noise_snr_db_range[1]:.1f} dB SNR")
            if dc_offset_range:
                aug_info.append(f"DC offset: ±{dc_offset_range:.4f}")
            
            logger.info(
                "Audio augmentation enabled: %s", ", ".join(aug_info)
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

        # Apply audio augmentations during training
        if (
            self.training
            and self.audio_augmentation_config
            and self.audio_augmentation_config.get("enabled", False)
        ):
            batch_size = audio.shape[0]
            device = audio.device
            
            # 1. Random gain augmentation
            gain_range_db = self.audio_augmentation_config.get("gain_range_db", 3.0)
            if gain_range_db > 0:
                # Sample random gain for each sample in the batch
                random_gains_db = torch.empty(
                    batch_size, device=device
                ).uniform_(-gain_range_db, gain_range_db)
                # Convert dB to linear: gain_linear = 10^(gain_db / 20)
                random_gains_linear = torch.pow(10.0, random_gains_db / 20.0)
                # Apply gain: expand dimensions to match audio shape [batch, samples]
                if audio.dim() == 2:
                    random_gains_linear = random_gains_linear.unsqueeze(1)
                audio = audio * random_gains_linear
            
            # 2. Noise injection
            noise_snr_db_range = self.audio_augmentation_config.get("noise_snr_db_range", None)
            if noise_snr_db_range is not None:
                # Sample random SNR for each sample in the batch
                snr_db = torch.empty(
                    batch_size, device=device
                ).uniform_(noise_snr_db_range[0], noise_snr_db_range[1])
                
                # Calculate signal RMS for each sample (audio is guaranteed to be at least 2D here)
                # Shape: [batch_size, num_samples] -> [batch_size, 1]
                signal_rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
                
                # Calculate noise level based on SNR: noise_rms = signal_rms / 10^(SNR/20)
                noise_rms = signal_rms / torch.pow(10.0, snr_db.unsqueeze(-1) / 20.0)
                
                # Generate white noise
                noise = torch.randn_like(audio)
                # Normalize noise to have RMS = 1, then scale to desired level
                noise_rms_actual = torch.sqrt(torch.mean(noise ** 2, dim=-1, keepdim=True))
                noise = noise / (noise_rms_actual + 1e-8) * noise_rms
                
                # Add noise to audio
                audio = audio + noise
            
            # 3. DC offset
            dc_offset_range = self.audio_augmentation_config.get("dc_offset_range", None)
            if dc_offset_range is not None and dc_offset_range > 0:
                # Sample random DC offset for each sample in the batch
                dc_offset = torch.empty(
                    batch_size, device=device
                ).uniform_(-dc_offset_range, dc_offset_range)
                # Expand dimensions to match audio shape [batch_size, num_samples]
                dc_offset = dc_offset.unsqueeze(-1)
                audio = audio + dc_offset
                
                # Clip to prevent overflow (audio should be in [-1, 1] range)
                audio = torch.clamp(audio, -1.0, 1.0)

        batch_size = audio.shape[0]
        encoded_list = []
        for i in range(batch_size):
            single_audio = audio[i]
            encoded = self.audio_encoder(single_audio)
            if encoded.dim() == 3:
                encoded = encoded.squeeze(0)
            encoded_list.append(encoded)
        return torch.stack(encoded_list, dim=0)

    @torch.no_grad()
    def decode_audio_embeddings_to_tokens(
        self,
        audio_embeddings: torch.Tensor,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """
        Decode projected audio embeddings to the nearest tokenizer vocabulary tokens.

        Args:
            audio_embeddings: Tensor shaped [batch, time, hidden_size] in LLM space.
            top_k: Number of nearest tokens to return per audio embedding.

        Returns:
            Dictionary with token ids, cosine similarities, and decoded token strings.
        """
        if audio_embeddings.dim() != 3:
            raise ValueError(
                f"audio_embeddings must be 3D [batch, time, hidden], got {audio_embeddings.shape}"
            )

        embedding_layer = self.llm.get_input_embeddings()
        vocab_embeddings = embedding_layer.weight  # [vocab, hidden]
        vocab_size, hidden_size = vocab_embeddings.shape
        batch_size, time_steps, embed_dim = audio_embeddings.shape

        if embed_dim != hidden_size:
            raise ValueError(
                f"audio_embeddings hidden dim ({embed_dim}) must match LLM hidden size ({hidden_size})"
            )

        top_k = max(1, min(top_k, vocab_size))

        # Align dtype/device for similarity computation
        audio_embeddings = audio_embeddings.to(
            device=vocab_embeddings.device, dtype=vocab_embeddings.dtype
        )

        audio_flat = audio_embeddings.reshape(-1, hidden_size)

        audio_norm = torch.nn.functional.normalize(audio_flat, p=2, dim=-1)
        vocab_norm = torch.nn.functional.normalize(vocab_embeddings, p=2, dim=-1)

        similarities = torch.matmul(audio_norm, vocab_norm.transpose(0, 1))
        top_similarities, top_indices = torch.topk(
            similarities, k=top_k, dim=-1, largest=True, sorted=True
        )

        top_similarities = top_similarities.view(batch_size, time_steps, top_k)
        top_indices = top_indices.view(batch_size, time_steps, top_k)

        decoded_tokens: List[List[List[str]]] = []
        for batch_idx in range(batch_size):
            batch_tokens: List[List[str]] = []
            for time_idx in range(time_steps):
                token_candidates = []
                for rank in range(top_k):
                    token_id = top_indices[batch_idx, time_idx, rank].item()
                    decoded = self.tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    token_candidates.append(decoded)
                batch_tokens.append(token_candidates)
            decoded_tokens.append(batch_tokens)

        return {
            "token_ids": top_indices,
            "similarities": top_similarities,
            "decoded_tokens": decoded_tokens,
        }

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        audio = kwargs.get("audio")
        labels = kwargs.get("labels")
        anchor_audio = kwargs.get("anchor_audio", None)
        mix_audio = kwargs.get("mix_audio", None)

        # Handle cross-attention projection which needs anchor and mix separately
        if self.use_cross_attention:
            if anchor_audio is not None and mix_audio is not None:
                # Use provided anchor and mix audio
                anchor_features = self.encode_audio(anchor_audio)
                mix_features = self.encode_audio(mix_audio)
                anchor_features = anchor_features.to(
                    dtype=next(self.audio_projection.parameters()).dtype
                )
                mix_features = mix_features.to(
                    dtype=next(self.audio_projection.parameters()).dtype
                )
                projected_audio_embeds = self.audio_projection(
                    anchor_features=anchor_features, mix_features=mix_features
                )
            else:
                # Fallback: use the same audio for both anchor and mix
                # This is a temporary solution until dataset provides them separately
                audio_features = self.encode_audio(audio)
                audio_features = audio_features.to(
                    dtype=next(self.audio_projection.parameters()).dtype
                )
                projected_audio_embeds = self.audio_projection(
                    anchor_features=audio_features, mix_features=audio_features
                )
        else:
            # Standard projection (linear or MLP)
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
        prefix_tokens: Optional[torch.Tensor] = None,
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

        # Handle cross-attention projection in generate method
        if self.use_cross_attention:
            # For generation, use the same audio for both anchor and mix
            # (In future, could accept anchor_audio and mix_audio as parameters)
            audio_features = self.encode_audio(audio)
            audio_features = audio_features.to(
                dtype=next(self.audio_projection.parameters()).dtype
            )
            projected_audio_embeds = self.audio_projection(
                anchor_features=audio_features, mix_features=audio_features
            )
        else:
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

        # If prefix tokens are provided, add them to the input
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.to(device)
            prefix_embeds = self.llm.get_input_embeddings()(prefix_tokens)
            prefix_embeds = prefix_embeds.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((inputs_embeds, prefix_embeds), dim=1)
            prefix_attention_mask = torch.ones(
                prefix_tokens.shape[:2], dtype=torch.long, device=device
            )
            attention_mask = torch.cat((attention_mask, prefix_attention_mask), dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
