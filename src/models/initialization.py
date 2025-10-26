import logging
from typing import Tuple

import torch
from omegaconf import DictConfig

from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.model_utils import (
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    initialize_tokenizer,
)


logger = logging.getLogger(__name__)


def initialize_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[ModularMultimodalModel, object]:
    """Initialize model, tokenizer, and LoRA configuration."""
    logger.info("Initializing model and tokenizer...")
    tokenizer = initialize_tokenizer(cfg.model.model_name)

    target_modules = cfg.model.lora.get("target_modules", [])
    use_lora = len(target_modules) > 0

    if not use_lora:
        logger.info(
            "Projection-only training mode: LLM frozen; training audio projection only"
        )
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if cfg.model.use_qlora:
            logger.info(
                "Loading base model with QLoRA quantization (no LoRA adapters)..."
            )
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=cfg.model.quantization.load_in_4bit,
                bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(
                    torch, cfg.model.quantization.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
            )
            llm = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                torch_dtype="auto",
                quantization_config=quantization_config,
            )
        else:
            logger.info("Loading base model without quantization (no LoRA adapters)...")
            llm = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                torch_dtype="auto",
            )

        for param in llm.parameters():
            param.requires_grad = False

        if hasattr(llm, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing for memory efficiency...")
            llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        lora_config = create_lora_config(cfg)
        if cfg.model.use_qlora:
            llm = initialize_qlora_model(cfg, lora_config, tokenizer)
        else:
            llm = initialize_lora_model(cfg, lora_config, tokenizer)

    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),
        projection_config=cfg.model.get("projection"),
    )
    logger.info("Model and tokenizer initialized.")
    model.print_trainable_parameters()
    return model, tokenizer
