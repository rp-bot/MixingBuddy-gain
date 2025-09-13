"""
LoRA (Low-Rank Adaptation) model implementation for LLM fine-tuning.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class LoRAModel:
    """LoRA model wrapper for efficient fine-tuning."""

    def __init__(self, config: DictConfig):
        """
        Initialize LoRA model.

        Args:
            config: Model configuration from Hydra
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def load_model(self) -> PreTrainedModel:
        """Load the base model with optional quantization."""
        logger.info(f"Loading model: {self.config.pretrained_model_name_or_path}")

        # Set up quantization config if specified
        quantization_config = None
        if self.config.get("quantization", {}).get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, self.config.quantization.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            )
        elif self.config.get("quantization", {}).get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.pretrained_model_name_or_path,
            "trust_remote_code": self.config.get("trust_remote_code", False),
            "torch_dtype": (
                getattr(torch, self.config.get("torch_dtype", "auto"))
                if self.config.get("torch_dtype") != "auto"
                else "auto"
            ),
            "device_map": self.config.get("device_map", "auto"),
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Use the correct model class for different models
        model_name = self.config.pretrained_model_name_or_path
        if "qwen2-audio" in model_name.lower() or "qwen2_audio" in model_name.lower():
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Prepare model for k-bit training if quantized
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model loaded successfully")
        return self.model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.pretrained_model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            trust_remote_code=self.config.get("trust_remote_code", False),
            use_fast=True,
            padding_side="right",
            truncation_side="right",
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Tokenizer loaded successfully")
        return self.tokenizer

    def setup_lora(self) -> PeftModel:
        """Set up LoRA configuration and apply to model."""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up LoRA")

        logger.info("Setting up LoRA configuration")

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=self.config.lora.inference_mode,
        )

        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.peft_model.print_trainable_parameters()

        logger.info("LoRA setup completed")
        return self.peft_model

    def get_model(self) -> Union[PreTrainedModel, PeftModel]:
        """Get the model (with LoRA if applied)."""
        if self.peft_model is not None:
            return self.peft_model
        elif self.model is not None:
            return self.model
        else:
            raise ValueError("Model not loaded")

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return self.tokenizer

    def save_model(self, output_dir: Union[str, Path], save_tokenizer: bool = True):
        """Save the model and optionally the tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self.get_model()

        if isinstance(model, PeftModel):
            # Save LoRA adapters
            model.save_pretrained(output_dir)
            logger.info(f"Saved LoRA adapters to {output_dir}")
        else:
            # Save full model
            model.save_pretrained(output_dir)
            logger.info(f"Saved full model to {output_dir}")

        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved tokenizer to {output_dir}")

    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if self.model is None:
            self.load_model()

        if self.peft_model is None:
            self.setup_lora()

        # Load LoRA weights
        self.peft_model = PeftModel.from_pretrained(
            self.model, checkpoint_path, torch_dtype=torch.float16
        )

        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

    def merge_and_unload(self) -> PreTrainedModel:
        """Merge LoRA weights with base model and unload LoRA."""
        if self.peft_model is None:
            raise ValueError("No LoRA model to merge")

        logger.info("Merging LoRA weights with base model")
        merged_model = self.peft_model.merge_and_unload()

        return merged_model

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        if self.peft_model is None:
            return {"total": 0, "trainable": 0, "percentage": 0.0}

        trainable_params = 0
        all_param = 0

        for _, param in self.peft_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "total": all_param,
            "trainable": trainable_params,
            "percentage": 100 * trainable_params / all_param if all_param > 0 else 0.0,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using the model."""
        model = self.get_model()

        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )

        return outputs


def create_lora_model(config: DictConfig) -> LoRAModel:
    """Factory function to create a LoRA model."""
    model = LoRAModel(config)
    model.load_model()
    model.load_tokenizer()
    model.setup_lora()
    return model
