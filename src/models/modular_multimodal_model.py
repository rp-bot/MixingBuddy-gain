import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to access src module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.encoders.encodec import EncodecEncoder


class ModularMultimodalModel(nn.Module):
    """
    A modular multimodal model that can be used for fine-tuning and inference.
    It is designed to be extended with different modalities.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        super().__init__()
        self.model_name = model_name
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm.config.pad_token_id = self.llm.config.eos_token_id

        # Initialize audio encoder (frozen EnCodec)
        self.audio_encoder = EncodecEncoder(freeze=True)

        # Add projection layer
        llm_hidden_size = self.llm.config.hidden_size
        audio_hidden_size = self.audio_encoder.output_dim
        self.audio_projection = nn.Linear(audio_hidden_size, llm_hidden_size)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio using the frozen EnCodec model.

        Args:
            audio (torch.Tensor): Audio tensor of shape (batch_size, samples) or (samples,)

        Returns:
            torch.Tensor: Encoded audio features
        """
        return self.audio_encoder(audio)

    def forward(self, input_ids, attention_mask, audio=None, labels=None):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input IDs for the language model.
            attention_mask (torch.Tensor): The attention mask for the language model.
            audio (torch.Tensor, optional): Audio input to be processed.
            labels (torch.Tensor, optional): The labels for computing the loss. Defaults to None.

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: The output from the language model.
        """
        output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
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
            # if audio.dim() == 1:
            #     audio = audio.unsqueeze(0)
            audio = audio.to(device)

            audio_features = self.encode_audio(audio)
            projected_audio_embeds = self.audio_projection(audio_features)
            print(f"Projected audio embeds shape: {projected_audio_embeds.shape}")
            text_embeds = self.llm.model.embed_tokens(model_inputs.input_ids)

            # Ensure projected audio embeddings match the LLM's dtype
            projected_audio_embeds = projected_audio_embeds.to(text_embeds.dtype)

            inputs_embeds = torch.cat((projected_audio_embeds, text_embeds), dim=1)
            print(f"Inputs embeds shape: {inputs_embeds.shape}")
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
            prompt_length = inputs_embeds.shape[1]
        else:
            generate_kwargs = {"input_ids": model_inputs.input_ids}
            prompt_length = model_inputs.input_ids.shape[1]

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
    model = ModularMultimodalModel()

    # Move the model to a CUDA device if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model moved to {device}")

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
        f"Audio shapes - anchor: {sample['anchor_audio'].shape}, mix: {sample['mix_audio'].shape}"
    )

    # Test generation with audio
    print("\n--- Testing generation with audio ---")
    instruction = sample["instruction"]
    mix_audio = sample["mix_audio"]

    generated_text_with_audio = model.generate(
        text_input=instruction, audio=mix_audio, max_new_tokens=150
    )
    print(f"Instruction: {instruction}")
    print(f"Generated response (with audio): {generated_text_with_audio}")
    print("-----------------------------------")

    # Test generation without audio
    print("\n--- Testing generation without audio ---")
    prompt_text = "What are the benefits of using a modular approach in deep learning?"
    print(f"Generating response for prompt: '{prompt_text}'")
    generated_text_no_audio = model.generate(prompt_text, max_new_tokens=150)

    # Print the response
    print("\n--- Model Response (no audio) ---")
    print(generated_text_no_audio)
    print("---------------------------------")
