import torch
import torch.nn as nn
from typing import Optional, Any

# Add parent directory to path to access src module
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.encoders.encodec import EncodecEncoder


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
    ):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
            use_qlora (bool): Whether to use QLoRA quantization (handled by training script).
            lora_config (LoraConfig, optional): The LoRA configuration to use (handled by training script).
            llm: Pre-configured language model (required).
            tokenizer: Pre-configured tokenizer (required).
        """
        super().__init__()
        self.model_name = model_name

        # Use pre-configured LLM and tokenizer (always provided by training script)
        self.llm = llm
        self.tokenizer = tokenizer

        # Note: QLoRA and LoRA setup is now handled by the training script
        # This keeps the model constructor focused on basic model initialization

        # Initialize audio encoder (frozen EnCodec)
        self.audio_encoder = EncodecEncoder(freeze=True)

        # Add projection layer
        llm_hidden_size = self.llm.config.hidden_size
        audio_hidden_size = self.audio_encoder.output_dim
        self.audio_projection = nn.Linear(audio_hidden_size, llm_hidden_size)

        # Move other modules to the same device as the LLM
        # PEFT with quantization handles the device placement of the LLM,
        # so we need to manually move the other modules.
        llm_device = self.llm.device
        self.audio_encoder.to(llm_device)
        self.audio_projection.to(llm_device)

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )

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
            squeeze_output = True
        else:
            squeeze_output = False

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

        # If input was unbatched, remove batch dimension
        if squeeze_output:
            batched_features = batched_features.squeeze(0)

        return batched_features

    def forward(self, input_ids, attention_mask, audio, labels, **kwargs):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input IDs for the language model.
            attention_mask (torch.Tensor): The attention mask for the language model.
            audio (torch.Tensor, optional): Audio input to be processed.
            labels (torch.Tensor, optional): The labels for computing the loss. Defaults to None.
            **kwargs: Additional arguments (metadata) that are ignored.

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: The output from the language model.
        """
        # 1. Encode audio
        audio_features = self.encode_audio(audio)

        # 2. Project audio features to LLM's embedding space
        projected_audio_embeds = self.audio_projection(audio_features)

        # 3. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 4. Ensure dtypes match
        projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)

        # 5. Concatenate audio and text embeddings
        inputs_embeds = torch.cat((projected_audio_embeds, text_embeds), dim=1)

        # 6. Create attention mask for audio
        audio_attention_mask = torch.ones(
            projected_audio_embeds.shape[:2],
            dtype=torch.long,
            device=input_ids.device,
        )

        # 7. Concatenate attention masks
        attention_mask = torch.cat((audio_attention_mask, attention_mask), dim=1)

        # 8. Adjust labels for audio part (ignore in loss)
        if labels is None:
            raise ValueError("labels cannot be None")

        audio_labels = torch.full(
            projected_audio_embeds.shape[:2],
            -100,
            dtype=torch.long,
            device=labels.device,
        )
        labels = torch.cat((audio_labels, labels), dim=1)

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,  # Disable cache to avoid padding issues
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        text_input: str,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generates text based on a given prompt using the underlying model's
        generate method. This is for inference only.

        Args:
            text_input (str): The prompt to generate text from.
            audio (torch.Tensor, optional): Audio input to be processed.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        device = self.llm.device
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text_input},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

        if audio is not None:
            audio = audio.to(device)

            audio_features = self.encode_audio(audio)
            projected_audio_embeds = self.audio_projection(audio_features)
            text_embeds = self.llm.get_input_embeddings()(model_inputs.input_ids)

            # Ensure projected audio embeddings match the LLM's dtype
            projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)

            inputs_embeds = torch.cat((projected_audio_embeds, text_embeds), dim=1)
            audio_attention_mask = torch.ones(
                projected_audio_embeds.shape[:2],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.cat(
                (audio_attention_mask, model_inputs.attention_mask), dim=1
            )

            generate_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
        else:
            generate_kwargs = {"input_ids": model_inputs.input_ids}

        generated_ids = self.llm.generate(
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
        )

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
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
        text_input=instruction, audio=mix_audio, max_new_tokens=150
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
    generated_text_no_audio = model.generate(prompt_text, max_new_tokens=150)

    # Print the response
    print("\n--- Model Response (no audio) ---")
    print(generated_text_no_audio)
    print("---------------------------------")
