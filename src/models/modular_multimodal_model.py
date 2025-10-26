import torch
import torch.nn as nn
from typing import Optional, Any, Dict

# Add parent directory to path to access src module
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.encoders.encodec import EncodecEncoder
from src.models.projections import LinearProjection, MLPProjection


class ModularMultimodalModel(nn.Module):
    """
    A modular multimodal model that can be used for fine-tuning and inference.
    It is designed to be extended with different modalities.
    """

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
        """
        Initializes the model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            use_qlora (bool): Whether to use QLoRA quantization (handled by training script).
            lora_config (LoraConfig, optional): The LoRA configuration to use (handled by training script).
            llm: Pre-configured language model (required).
            tokenizer: Pre-configured tokenizer (required).
            encoder_config (dict, optional): Configuration for the audio encoder.
            projection_config (dict, optional): Configuration for the audio projection layer.
        """
        super().__init__()
        self.model_name = model_name

        # Use pre-configured LLM and tokenizer (always provided by training script)
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = llm.config  # Expose the llm's config for SFTTrainer

        # Note: QLoRA and LoRA setup is now handled by the training script
        # This keeps the model constructor focused on basic model initialization

        # Initialize audio encoder with configuration
        if encoder_config is None:
            # Default encoder configuration
            encoder_config = {
                "model_name": "facebook/encodec_24khz",
                "freeze": True,
                "device": None,
            }

        # Set device to match LLM device if not specified
        if encoder_config.get("device") is None:
            encoder_config["device"] = self.llm.device

        self.audio_encoder = EncodecEncoder(**encoder_config)

        # Initialize projection layer based on configuration
        llm_hidden_size = self.llm.config.hidden_size
        audio_hidden_size = self.audio_encoder.output_dim

        # Extract auxiliary loss configuration
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
            # Default to simple linear projection for backward compatibility
            self.audio_projection = nn.Linear(audio_hidden_size, llm_hidden_size)
        elif projection_config.get("type") == "mlp":
            # Use MLP projection
            # Convert DictConfig to regular dict to avoid struct mode issues
            mlp_config = dict(projection_config)
            mlp_config.pop("type", None)  # Remove type from config
            mlp_config.pop("use_auxiliary_loss", None)  # Remove auxiliary loss config
            mlp_config.pop("auxiliary_loss_weight", None)
            self.audio_projection = MLPProjection(
                input_dim=audio_hidden_size, output_dim=llm_hidden_size, **mlp_config
            )
        else:
            raise ValueError(
                f"Unsupported projection type: {projection_config.get('type')}"
            )

        # Move other modules to the same device as the LLM
        # PEFT with quantization handles the device placement of the LLM,
        # so we need to manually move the other modules.
        llm_device = self.llm.device
        self.audio_encoder.to(llm_device)
        self.audio_projection.to(llm_device)

        # Print auxiliary loss configuration
        if self.use_auxiliary_loss:
            print(
                f"✓ Auxiliary loss enabled with weight {self.auxiliary_loss_weight} "
                f"to improve projection layer training"
            )

    def print_trainable_parameters(self):
        """
        Prints detailed parameter information for the model.
        """
        print("=" * 80)
        print("MODEL PARAMETER ANALYSIS")
        print("=" * 80)

        # Overall statistics
        trainable_params = 0
        all_param = 0
        frozen_params = 0

        # Component-wise breakdown
        llm_params = 0
        llm_trainable = 0
        audio_encoder_params = 0
        audio_encoder_trainable = 0
        audio_projection_params = 0
        audio_projection_trainable = 0

        print("\n📊 COMPONENT BREAKDOWN:")
        print("-" * 50)

        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()

            # Categorize by component
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

        # Print component details
        print(f"🤖 LLM (Language Model):")
        print(f"   Total: {llm_params:,} params ({llm_params / 1e6:.2f}M)")
        print(f"   Trainable: {llm_trainable:,} params ({llm_trainable / 1e6:.2f}M)")
        print(
            f"   Frozen: {llm_params - llm_trainable:,} params ({(llm_params - llm_trainable) / 1e6:.2f}M)"
        )
        print(f"   Trainable %: {100 * llm_trainable / llm_params:.1f}%")

        print(f"\n🎵 Audio Encoder (EnCodec):")
        print(
            f"   Total: {audio_encoder_params:,} params ({audio_encoder_params / 1e6:.2f}M)"
        )
        print(
            f"   Trainable: {audio_encoder_trainable:,} params ({audio_encoder_trainable / 1e6:.2f}M)"
        )
        print(
            f"   Frozen: {audio_encoder_params - audio_encoder_trainable:,} params ({(audio_encoder_params - audio_encoder_trainable) / 1e6:.2f}M)"
        )
        print(
            f"   Trainable %: {100 * audio_encoder_trainable / audio_encoder_params:.1f}%"
        )

        print(f"\n🔗 Audio Projection:")
        print(
            f"   Total: {audio_projection_params:,} params ({audio_projection_params / 1e6:.2f}M)"
        )
        print(
            f"   Trainable: {audio_projection_trainable:,} params ({audio_projection_trainable / 1e6:.2f}M)"
        )
        print(
            f"   Frozen: {audio_projection_params - audio_projection_trainable:,} params ({(audio_projection_params - audio_projection_trainable) / 1e6:.2f}M)"
        )
        print(
            f"   Trainable %: {100 * audio_projection_trainable / audio_projection_params:.1f}%"
        )

        print("\n📈 OVERALL STATISTICS:")
        print("-" * 50)
        print(f"Total Parameters: {all_param:,} ({all_param / 1e6:.2f}M)")
        print(
            f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)"
        )
        print(f"Frozen Parameters: {frozen_params:,} ({frozen_params / 1e6:.2f}M)")
        print(f"Trainable Percentage: {100 * trainable_params / all_param:.2f}%")

        # Memory estimation (rough)
        # Assuming float32 (4 bytes per parameter) for trainable params
        trainable_memory_mb = (trainable_params * 4) / (1024 * 1024)
        total_memory_mb = (all_param * 4) / (1024 * 1024)

        print(f"\n💾 MEMORY ESTIMATION:")
        print("-" * 50)
        print(f"Trainable Parameters Memory: {trainable_memory_mb:.1f} MB")
        print(f"Total Model Memory: {total_memory_mb:.1f} MB")

        # Training efficiency metrics
        efficiency_ratio = trainable_params / all_param if all_param > 0 else 0
        print(f"\n⚡ TRAINING EFFICIENCY:")
        print("-" * 50)
        print(f"Parameter Efficiency: {efficiency_ratio:.3f}")
        if efficiency_ratio > 0.1:
            print("   ⚠️  High trainable ratio - consider more aggressive LoRA settings")
        elif efficiency_ratio < 0.01:
            print("   ✅ Very efficient - good LoRA configuration")
        else:
            print("   ✅ Good balance between efficiency and capacity")

        print("=" * 80)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio using the frozen EnCodec model.

        Args:
            audio (torch.Tensor): Audio tensor of shape (batch_size, samples) or (samples,)

        Returns:
            torch.Tensor: Encoded audio features of shape (batch_size, seq_len, hidden_dim)
        """
        # Handle both batched and unbatched input
        if audio.dim() == 1:
            # Single sample: [samples] → add batch dim
            audio = audio.unsqueeze(0)

        # Now audio is [batch_size, samples]
        # Process each sample in the batch individually (EnCodec expects 1D input)
        batch_size = audio.shape[0]
        encoded_list = []

        for i in range(batch_size):
            # Extract single sample: [samples]
            single_audio = audio[i]
            # Encode it: returns [1, seq_len, hidden_dim] or [seq_len, hidden_dim]
            encoded = self.audio_encoder(single_audio)
            # Remove the extra dimension if present
            if encoded.dim() == 3:
                encoded = encoded.squeeze(
                    0
                )  # [1, seq_len, hidden_dim] → [seq_len, hidden_dim]
            encoded_list.append(encoded)

        # Stack into batch: [batch_size, seq_len, hidden_dim]
        batched_features = torch.stack(encoded_list, dim=0)

        return batched_features

    def forward(self, **kwargs):
        """
        Defines the forward pass of the model.

        Args:
            **kwargs: A dictionary of arguments that must contain:
                - input_ids (torch.Tensor): The input IDs for the language model.
                - attention_mask (torch.Tensor): The attention mask for the language model.
                - audio (torch.Tensor): Audio input to be processed.
                - labels (torch.Tensor): The labels for computing the loss.

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: The output from the language model.
        """
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        audio = kwargs.get("audio")
        labels = kwargs.get("labels")

        # 1. Encode audio
        audio_features = self.encode_audio(audio)

        # 2. Project audio features to LLM's embedding space
        projected_audio_embeds = self.audio_projection(audio_features)

        # 3. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 4. Compute auxiliary loss if enabled (to ensure strong gradient flow to projection)
        auxiliary_loss = torch.tensor(0.0, device=projected_audio_embeds.device)
        if self.use_auxiliary_loss and self.training:
            # Compute the mean L2 norm per embedding dimension
            # This keeps the loss magnitude reasonable
            audio_norm_per_dim = (
                torch.norm(projected_audio_embeds, p=2, dim=-1)
                / projected_audio_embeds.shape[-1]
            )
            text_norm_per_dim = (
                torch.norm(text_embeds, p=2, dim=-1) / text_embeds.shape[-1]
            )

            # L2 loss to match normalized norms
            auxiliary_loss = torch.nn.functional.mse_loss(
                audio_norm_per_dim.mean(), text_norm_per_dim.mean().detach()
            )

        # 5. Replace the dummy audio token embeddings with the projected audio embeddings
        num_audio_tokens = projected_audio_embeds.shape[1]

        # Ensure dtypes match before concatenation
        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)

        # Concatenate the projected audio embeddings with the actual text embeddings
        inputs_embeds = torch.cat(
            [projected_audio_embeds, text_embeds[:, num_audio_tokens:]], dim=1
        )

        # The collator now prepares the correct attention_mask and labels,
        # so we no longer need to modify them here. The SFTTrainer will receive
        # inputs with consistent sequence lengths.
        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,  # Disable cache to avoid padding issues
        )

        # Add auxiliary loss to the main loss
        if self.use_auxiliary_loss and self.training:
            weighted_aux_loss = self.auxiliary_loss_weight * auxiliary_loss
            output.loss = output.loss + weighted_aux_loss
            # Store auxiliary loss for logging (optional, won't break if not used)
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
        """
        Generates text based on a given prompt using the underlying model's
        generate method. This is for inference only.

        Args:
            text_input (str): The user's instruction or prompt.
            audio (torch.Tensor): Audio input to be processed.
            max_new_tokens (int): The maximum number of new tokens to generate.
            system_message (str, optional): The system message to guide the model.

        Returns:
            str: The generated text.
        """
        device = self.llm.device

        # Create messages in conversational format
        messages = []
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text_input})

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        audio = audio.to(device)

        # Encode and project audio
        audio_features = self.encode_audio(audio)
        projected_audio_embeds = self.audio_projection(audio_features)

        # Get text embeddings and combine with audio
        text_embeds = self.llm.get_input_embeddings()(model_inputs.input_ids)
        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat((projected_audio_embeds, text_embeds), dim=1)

        # Create combined attention mask
        audio_attention_mask = torch.ones(
            projected_audio_embeds.shape[:2], dtype=torch.long, device=device
        )
        attention_mask = torch.cat(
            (audio_attention_mask, model_inputs.attention_mask), dim=1
        )

        # Generate response
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

        # Decode and return the full response
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response


if __name__ == "__main__":
    import warnings
    import logging
    from typing import Optional
    from src.data.dataset import MixingDataset

    # Suppress warnings
    # The "resume_download" warning is a FutureWarning from huggingface_hub
    warnings.filterwarnings("ignore", category=FutureWarning)
    # The "special tokens" warning is logged by the transformers library
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # This block will only run when the script is executed directly
    # It serves as a quick test for the model's generation capabilities.
    print("Testing ModularMultimodalModel...")

    # Instantiate the model
    # The model will be downloaded from Hugging Face Hub the first time it's used.
    model = ModularMultimodalModel(use_qlora=True)

    # Move the model to a CUDA device if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device) # PEFT with quantization handles device placement
    print(f"Model loaded on {model.llm.device}")

    print("\n--- Trainable Parameters ---")
    model.print_trainable_parameters()
    print("--------------------------\n")

    # To verify target_modules, you can uncomment the following lines:
    # print("\n--- Model Modules ---")
    # with open("model_modules.txt", "w") as f:
    #     for name, module in model.llm.named_modules():
    #         if "proj" in name:
    #             f.write(name + "\n")
    #             print(name)
    print("---------------------\n")

    # Load one sample from dataset
    dataset = MixingDataset(
        jsonl_path="data/musdb18hq_processed/train/training_samples.jsonl",
        audio_root="data/musdb18hq_processed/train/flawed_mixes",
        tokenizer=model.tokenizer,
        sample_rate=24000,
        limit=1,
    )
    sample = dataset[0]
    print(f"Loaded sample: {sample['instruction'][:100]}...")
    print(
        f"Audio shapes - anchor: {sample['anchor_audio'].shape}, mix: {sample['audio'].shape}"
    )

    # Test generation with audio
    print("\n--- Testing generation with audio ---")
    instruction = sample["instruction"]
    mix_audio = sample["audio"]

    generated_text_with_audio = model.generate(
        text_input=instruction,
        audio=mix_audio,
        max_new_tokens=150,
        system_message="You are a helpful mixing assistant.",
    )
    print(f"Instruction: {instruction}")
    print(f"Generated response (with audio): {generated_text_with_audio}")
    print("-----------------------------------")

    # --- Test Training Step ---
    print("\n--- Testing training step ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Prepare batch
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    mix_audio = sample["audio"].to(device)
    labels = sample["labels"].unsqueeze(0).to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio=mix_audio,
        labels=labels,
    )
    loss = outputs.loss

    print(f"Loss: {loss.item()}")

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print("Training step completed successfully.")
    print("---------------------------\n")

    # Test generation without audio
    print("\n--- Testing generation without audio ---")
    prompt_text = "What are the benefits of using a modular approach in deep learning?"
    print(f"Generating response for prompt: '{prompt_text}'")
    generated_text_no_audio = model.generate(
        text_input=prompt_text,
        audio=torch.randn(24000 * 2),  # Dummy audio
        max_new_tokens=150,
        system_message="You are a helpful AI assistant.",
    )

    # Print the response
    print("\n--- Model Response (no audio) ---")
    print(generated_text_no_audio)
    print("---------------------------------")
