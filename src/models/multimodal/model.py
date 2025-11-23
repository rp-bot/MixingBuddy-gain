import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.encoders.encodec import EncodecEncoder
from src.models.encoders.mert import MERTEncoder
from src.models.encoders.passt import PaSSTEncoder
from src.models.projections import MLPProjection
from src.models.projections.cross_attention import CrossAttentionProjection


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
        use_teacher_forcing: bool = True,
        autoregressive_training: bool = False,
        max_autoregressive_steps: int = 40,
        scheduled_sampling_strategy: str = "linear",
        teacher_forcing_start_ratio: float = 1.0,
        teacher_forcing_end_ratio: float = 0.0,
        scheduled_sampling_warmup_steps: int = 0,
        total_training_steps: int = 1000,
    ):
        super().__init__()
        self.model_name = model_name

        self.llm = llm
        self.tokenizer = tokenizer
        self.config = llm.config

        # Add attributes needed for HuggingFace Trainer compatibility
        self._keys_to_ignore_on_save = []
        
        # Teacher forcing control
        self.use_teacher_forcing = use_teacher_forcing
        
        # Autoregressive training
        self.autoregressive_training = autoregressive_training
        self.max_autoregressive_steps = max_autoregressive_steps
        
        # Scheduled sampling parameters
        self.scheduled_sampling_strategy = scheduled_sampling_strategy
        self.teacher_forcing_start_ratio = teacher_forcing_start_ratio
        self.teacher_forcing_end_ratio = teacher_forcing_end_ratio
        self.scheduled_sampling_warmup_steps = scheduled_sampling_warmup_steps
        self.total_training_steps = total_training_steps
        
        if autoregressive_training:
            logger.info(f"Model initialized with autoregressive_training=True, max_steps={max_autoregressive_steps}")
            logger.info("Note: Autoregressive training is slower but matches inference behavior")
        elif scheduled_sampling_strategy != "none":
            logger.info(f"Model initialized with scheduled sampling: {scheduled_sampling_strategy}")
            logger.info(f"Ratio decay: {teacher_forcing_start_ratio} -> {teacher_forcing_end_ratio} over {total_training_steps} steps")
        else:
            logger.info(f"Model initialized with use_teacher_forcing={use_teacher_forcing}")

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
        elif "passt" in encoder_model_name.lower():
            logger.info(f"Using PaSST encoder: {encoder_model_name}")
            if "input_sample_rate" not in encoder_config:
                encoder_config["input_sample_rate"] = 24000
            self.audio_encoder = PaSSTEncoder(**encoder_config)
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
        
        # Register forward call count as a buffer so it persists across checkpoints
        # This is critical for scheduled sampling to maintain correct TF ratio after resume
        self.register_buffer('_forward_call_count', torch.tensor(0, dtype=torch.long))

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
        """Encode audio with encoder; supports batched and unbatched input."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]

        # Apply random gain augmentation in +/-3 dB range before encoding (only during training)
        if self.training:
            gains_db = torch.empty(batch_size, device=audio.device, dtype=audio.dtype).uniform_(
                -3.0, 3.0
            )
            gains_linear = torch.pow(
                torch.tensor(10.0, device=audio.device, dtype=audio.dtype), gains_db / 20.0
            )
            audio = audio * gains_linear.view(batch_size, 1)

        # Check if encoder is frozen (no trainable parameters)
        # Use no_grad for frozen encoders to save memory and computation
        encoder_has_trainable_params = any(
            p.requires_grad for p in self.audio_encoder.parameters()
        )
        
        # Try batch processing first for better GPU utilization
        # The encoder (especially MERT) can handle batches efficiently
        try:
            if not encoder_has_trainable_params:
                # Encoder is frozen - use no_grad for efficiency
                with torch.no_grad():
                    encoded = self.audio_encoder(audio)  # [batch, time_steps, feature_dim]
            else:
                # Encoder has trainable parameters - need gradients
                encoded = self.audio_encoder(audio)  # [batch, time_steps, feature_dim]
            
            # Some encoders may return variable-length sequences
            # If all sequences have the same length, return directly
            if encoded.dim() == 3:
                return encoded
            
            # Handle case where encoder returns 2D output (shouldn't happen but be safe)
            if encoded.dim() == 2:
                return encoded.unsqueeze(0) if batch_size == 1 else encoded.unsqueeze(1)
                
        except (RuntimeError, ValueError) as e:
            # Fallback to sequential processing if batch processing fails
            # This handles encoders that don't support batching or have issues
            logger.warning(f"Batch audio encoding failed, falling back to sequential processing: {e}")
            encoded_list = []
            for i in range(batch_size):
                single_audio = audio[i]
                if not encoder_has_trainable_params:
                    # Encoder is frozen - use no_grad for efficiency
                    with torch.no_grad():
                        encoded = self.audio_encoder(single_audio)
                else:
                    # Encoder has trainable parameters - need gradients
                    encoded = self.audio_encoder(single_audio)
                if encoded.dim() == 3:
                    encoded = encoded.squeeze(0)
                encoded_list.append(encoded)
            
            # Handle variable-length sequences (e.g., from PaSST)
            # Pad all sequences to the same length before stacking
            if encoded_list:
                # Find the maximum sequence length
                max_seq_len = max(enc.shape[0] for enc in encoded_list)
                feature_dim = encoded_list[0].shape[-1]
                
                # Pad all sequences to max_seq_len
                padded_encoded_list = []
                for enc in encoded_list:
                    if enc.shape[0] < max_seq_len:
                        # Pad sequence dimension (time_steps)
                        padding_size = max_seq_len - enc.shape[0]
                        # Pad at the end: (0, 0) for feature_dim, (0, padding_size) for seq_len
                        padded = torch.nn.functional.pad(
                            enc, (0, 0, 0, padding_size), mode="constant", value=0.0
                        )
                        padded_encoded_list.append(padded)
                    else:
                        padded_encoded_list.append(enc)
                
                return torch.stack(padded_encoded_list, dim=0)
            else:
                return torch.stack(encoded_list, dim=0)
        
        return encoded

    def _autoregressive_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Perform true autoregressive forward pass using model's own predictions.
        
        This method implements non-teacher forcing by:
        1. Prefilling the prompt portion
        2. Generating response tokens one-by-one using greedy predictions
        3. Computing loss on each generated token
        4. Accumulating and averaging the loss
        
        Args:
            inputs_embeds: Combined audio and text embeddings [batch, seq_len, hidden_size]
            attention_mask: Attention mask for inputs [batch, seq_len]
            input_ids: Token IDs for the full sequence [batch, seq_len]
            labels: Target labels with -100 for ignored positions [batch, seq_len]
            
        Returns:
            CausalLMOutputWithPast containing loss, logits, and other outputs
        """
        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        seq_len = inputs_embeds.shape[1]
        
        # Find where the response starts for each sample (where labels switch from -100 to valid tokens)
        response_start_positions = []
        for i in range(batch_size):
            label_seq = labels[i]
            response_start = None
            for j in range(len(label_seq)):
                if label_seq[j] != -100:
                    response_start = j
                    break
            response_start_positions.append(response_start if response_start is not None else seq_len)
        
        # Find maximum response start to ensure we prefill all prompts fully
        # This handles cases where samples have different prompt lengths
        max_response_start = max(response_start_positions) if response_start_positions else seq_len
        min_response_start = min(response_start_positions) if response_start_positions else seq_len
        
        if min_response_start >= seq_len:
            # No response tokens, fall back to standard training
            logger.warning("No response tokens found, falling back to standard training")
            output = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
            return output
        
        # Prefill: process prompt portion up to max_response_start to ensure all prompts are fully processed
        # This handles variable-length prompts correctly
        prompt_embeds = inputs_embeds[:, :max_response_start]
        prompt_attention = attention_mask[:, :max_response_start] if attention_mask is not None else None
        
        prefill_output = self.llm(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_attention,
            return_dict=True,
            use_cache=True,
        )
        
        past_key_values = prefill_output.past_key_values
        accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        total_steps = 0
        
        # Get the starting token for each sample individually (last token before response starts)
        # This correctly handles variable-length prompts
        starting_token_ids = []
        for i in range(batch_size):
            response_start = response_start_positions[i]
            if response_start > 0:
                # Use the last token of the prompt (one before response starts)
                starting_token_ids.append(input_ids[i, response_start - 1].item())
            else:
                # Edge case: response starts immediately, use BOS token
                bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
                starting_token_ids.append(bos_token_id)
        
        # Convert to tensor for batch processing
        current_token_ids = torch.tensor(starting_token_ids, device=device, dtype=torch.long).unsqueeze(1)  # [batch, 1]
        
        # Collect logits from each autoregressive step for constructing full-sequence logits
        # This allows SFTTrainer to compute token accuracy correctly
        collected_logits = []
        
        # Autoregressive loop: generate tokens one by one using greedy predictions
        for step in range(self.max_autoregressive_steps):
            # Check if we have valid targets for this step
            # A sample is valid if it has a non-(-100) target token at the current position
            valid_samples = []
            targets_list = []
            
            for i in range(batch_size):
                response_start = response_start_positions[i]
                target_pos = response_start + step
                
                if target_pos < labels.shape[1]:
                    target_token = labels[i, target_pos].item()
                    if target_token != -100:
                        valid_samples.append(i)
                        targets_list.append(target_token)
            
            if not valid_samples:
                # All samples are done (no more valid targets)
                break
            
            # Forward pass with current token for all samples
            # Note: Even if some samples are finished, we still process them in the batch
            # for efficiency, but we only compute loss for valid samples
            current_embeds = self.llm.get_input_embeddings()(current_token_ids)
            current_attention = torch.ones_like(current_token_ids, device=device, dtype=torch.long)
            
            step_output = self.llm(
                inputs_embeds=current_embeds,
                attention_mask=current_attention,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            
            # Update past_key_values for next iteration
            past_key_values = step_output.past_key_values
            
            # Get logits for the last position (current token prediction)
            logits = step_output.logits[:, -1, :]  # [batch, vocab_size]
            
            # Collect logits for constructing full-sequence logits (for token accuracy computation)
            # Use clone().detach() to avoid gradient issues and memory problems
            collected_logits.append(logits.clone().detach())
            
            # Compute loss only for valid samples (those with ground truth targets)
            # This handles variable-length responses correctly
            if len(valid_samples) == batch_size:
                # All samples are valid - compute loss for all
                targets_tensor = torch.tensor(targets_list, device=device, dtype=torch.long)
                step_loss = loss_fct(logits, targets_tensor)
            else:
                # Only some samples are valid - compute loss only for those
                # This happens when some samples have finished generating
                valid_logits = logits[valid_samples]
                targets_tensor = torch.tensor(targets_list, device=device, dtype=torch.long)
                step_loss = loss_fct(valid_logits, targets_tensor)
            
            # Accumulate loss (will be averaged later)
            accumulated_loss = accumulated_loss + step_loss
            total_steps += 1
            
            # Get greedy predictions for next step (argmax over vocabulary)
            # Note: For finished samples, we still generate predictions but won't use them for loss
            predicted_tokens = logits.argmax(dim=-1)  # [batch]
            current_token_ids = predicted_tokens.unsqueeze(1)  # [batch, 1]
        
        # Average the loss across all steps
        # Since step_loss uses reduction='mean', each step_loss is already averaged over valid samples
        # Averaging across steps gives us the average loss per token across all generated tokens
        if total_steps > 0:
            final_loss = accumulated_loss / total_steps
        else:
            final_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Create output object
        # Construct full-sequence logits from collected autoregressive logits
        # This allows SFTTrainer to compute token accuracy correctly
        vocab_size = self.llm.config.vocab_size
        # Initialize full logits with zeros
        full_logits = torch.zeros(
            (batch_size, seq_len, vocab_size),
            device=device,
            dtype=torch.float32
        )
        
        # Fill response positions with logits from autoregressive steps
        for step_idx, step_logits in enumerate(collected_logits):
            for i in range(batch_size):
                response_start = response_start_positions[i]
                target_pos = response_start + step_idx
                # Only fill if this position has a valid target (not -100)
                if target_pos < seq_len and target_pos < labels.shape[1] and labels[i, target_pos] != -100:
                    full_logits[i, target_pos] = step_logits[i]
        
        # Detach to ensure logits don't affect the computation graph
        # (loss is already computed and gradients flow through final_loss)
        full_logits = full_logits.detach()
        
        output = CausalLMOutputWithPast(
            loss=final_loss,
            logits=full_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        
        if self._forward_call_count.item() <= 10:
            logger.info(f"Autoregressive loss computed: {final_loss.item():.4f} over {total_steps} steps (requires_grad={final_loss.requires_grad})")
        
        return output

    def get_teacher_forcing_ratio(self, current_step: int) -> float:
        """
        Calculate teacher forcing ratio based on current step and schedule strategy.
        """
        if self.scheduled_sampling_strategy == "fixed":
            # Always use start ratio (e.g. 0.5 for constant mixing)
            return self.teacher_forcing_start_ratio
            
        if self.scheduled_sampling_strategy == "none":
            # No scheduling, stick to configured static boolean
            return 1.0 if self.use_teacher_forcing else 0.0
            
        if current_step < self.scheduled_sampling_warmup_steps:
            return self.teacher_forcing_start_ratio
            
        # Calculate progress (0.0 to 1.0)
        # Avoid division by zero
        total_decay_steps = max(1, self.total_training_steps - self.scheduled_sampling_warmup_steps)
        progress = min(1.0, max(0.0, (current_step - self.scheduled_sampling_warmup_steps) / total_decay_steps))
        
        if self.scheduled_sampling_strategy == "linear":
            # Linear decay: start -> end
            ratio = self.teacher_forcing_start_ratio - progress * (self.teacher_forcing_start_ratio - self.teacher_forcing_end_ratio)
        elif self.scheduled_sampling_strategy == "cosine":
            # Cosine decay
            import math
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            ratio = self.teacher_forcing_end_ratio + (self.teacher_forcing_start_ratio - self.teacher_forcing_end_ratio) * cosine_decay
        else:
            # Default to linear
            ratio = self.teacher_forcing_start_ratio - progress * (self.teacher_forcing_start_ratio - self.teacher_forcing_end_ratio)
            
        return max(0.0, min(1.0, ratio))

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        audio = kwargs.get("audio")
        labels = kwargs.get("labels")
        anchor_audio = kwargs.get("anchor_audio", None)
        mix_audio = kwargs.get("mix_audio", None)
        # Allow override via kwargs, otherwise use instance attribute
        use_teacher_forcing = kwargs.get("use_teacher_forcing", self.use_teacher_forcing)
        
        # Increment forward call count (registered as buffer to persist across checkpoints)
        self._forward_call_count += 1
        if self._forward_call_count.item() == 1:
            logger.info(f"First forward call - use_teacher_forcing: {use_teacher_forcing}, training: {self.training}, labels: {labels is not None}")

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

        # Calculate expected number of audio tokens from audio length and encoder stride
        # This must match what the collator calculated: ceil(audio_length / hop_length)
        if hasattr(self.audio_encoder, 'hop_length'):
            audio_length = audio.shape[-1]  # Last dimension is the sequence length
            hop_length = self.audio_encoder.hop_length
            expected_num_audio_tokens = math.ceil(audio_length / hop_length)
        else:
            # Fallback: use actual number of tokens from projection
            expected_num_audio_tokens = projected_audio_embeds.shape[1]
        
        # Truncate or pad projected_audio_embeds to match expected_num_audio_tokens
        actual_num_tokens = projected_audio_embeds.shape[1]
        if actual_num_tokens > expected_num_audio_tokens:
            # Truncate to expected length
            projected_audio_embeds = projected_audio_embeds[:, :expected_num_audio_tokens, :]
        elif actual_num_tokens < expected_num_audio_tokens:
            # Pad to expected length
            padding_size = expected_num_audio_tokens - actual_num_tokens
            projected_audio_embeds = torch.nn.functional.pad(
                projected_audio_embeds, (0, 0, 0, padding_size), mode="constant", value=0.0
            )
        
        num_audio_tokens = expected_num_audio_tokens

        # Auxiliary loss: Critical for autoregressive training
        # In autoregressive mode, the projection layer only gets gradients from the prefill step
        # (not from the generation steps). The auxiliary loss provides direct gradient signal
        # to the projection layer by matching audio/text embedding norms.
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

        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)
        text_only_embeds = text_embeds[:, num_audio_tokens:]
        inputs_embeds = torch.cat([projected_audio_embeds, text_only_embeds], dim=1)

        # Determine mode (Teacher Forcing vs Autoregressive)
        use_autoregressive = False
        
        if self.training:
            if self.scheduled_sampling_strategy != "none":
                # Scheduled Sampling: Probabilistic switching
                # We use _forward_call_count as a proxy for "step" since we don't easily have global_step here
                # Note: This counts forward passes, which equals (steps * gradient_accumulation_steps)
                current_step = self._forward_call_count.item()
                tf_ratio = self.get_teacher_forcing_ratio(current_step)
                
                # Random choice
                use_tf = torch.rand(1).item() < tf_ratio
                use_autoregressive = not use_tf
                
                # Log occasionally
                if self._forward_call_count.item() % 100 == 0:
                    logger.info(f"Scheduled Sampling (Step {current_step}): TF Ratio={tf_ratio:.2f}, Using={'Teacher Forcing' if use_tf else 'Autoregressive'}")
            
            elif self.autoregressive_training:
                # Strict Autoregressive
                use_autoregressive = True
                
            elif not use_teacher_forcing:
                # Legacy/Manual disable
                use_autoregressive = True
        
        # Evaluation mode: use standard teacher forcing for faster, more stable loss computation
        # (Autoregressive generation is still used in model.generate() for actual inference)
        
        # Execute chosen path
        if use_autoregressive and labels is not None:
            if self._forward_call_count.item() <= 10:
                mode_str = "autoregressive_training" if self.autoregressive_training else "scheduled/manual"
                logger.info(f"Using true non-teacher forcing ({mode_str}): max_steps={self.max_autoregressive_steps}")
            
            output = self._autoregressive_forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                input_ids=input_ids,
                labels=labels,
            )
        else:
            # Standard forward pass with teacher forcing (default behavior)
            # This is also used during evaluation or when coin flip selects TF
            if self._forward_call_count.item() <= 10:
                logger.info(f"Using standard forward (Teacher Forcing): training={self.training}")
            
            output = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                use_cache=False,
            )

        # Add auxiliary loss if enabled and we have a loss
        # Note: This is especially important in autoregressive training where projection
        # layer gradients are small. Gradient scaling (via callback) can further amplify
        # these gradients if needed.
        if self.use_auxiliary_loss and self.training and hasattr(output, 'loss') and output.loss is not None:
            weighted_aux_loss = self.auxiliary_loss_weight * auxiliary_loss
            output.loss = output.loss + weighted_aux_loss
            output.auxiliary_loss = auxiliary_loss.item()

        return output

    @torch.no_grad()
    def generate(
        self,
        text_input: Optional[str] = None,
        audio: torch.Tensor = None,
        max_new_tokens: int = 256,
        system_message: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        **generate_kwargs: Any,
    ) -> str:
        device = self.llm.device

        if audio is None:
            raise ValueError("audio tensor must be provided for generation")

        if messages is None:
            if text_input is None:
                raise ValueError("Either messages or text_input must be provided.")
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

        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response
