import logging
import os
from typing import Tuple
from pathlib import Path

import torch
from omegaconf import DictConfig
from dotenv import load_dotenv

from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.model_utils import (
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    initialize_tokenizer,
)

# Load environment variables from .env file if it exists
# Look for .env in project root (2 levels up from src/models/)
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


logger = logging.getLogger(__name__)


def initialize_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[ModularMultimodalModel, object]:
    """Initialize model, tokenizer, and LoRA configuration."""
    import os
    logger.info("Initializing model and tokenizer...")
    
    # Get Hugging Face token from config or environment
    hf_token = cfg.model.get("hf_token") or os.getenv("HF_TOKEN")
    tokenizer = initialize_tokenizer(cfg.model.model_name, token=hf_token)

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
                token=hf_token,
            )
        else:
            logger.info("Loading base model without quantization (no LoRA adapters)...")
            llm = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_name,
                torch_dtype="auto",
                token=hf_token,
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
        audio_augmentation_config=cfg.training.get("audio_augmentation"),
    )
    
    # Handle frozen projection and encoder weights loading
    # Note: For Q-Former projections, weights can be loaded via pretrained_path in projection config
    # so we only require training.resume.checkpoint_path for non-Q-Former projections
    freeze_projection = cfg.model.projection.get("freeze_projection", False)
    freeze_layer_weights = cfg.model.encoder.get("freeze_layer_weights", False)
    projection_type = cfg.model.projection.get("type", "linear")
    
    # Check if Q-Former already loaded weights via pretrained_path
    qformer_pretrained = cfg.model.projection.get("pretrained_path", None)
    mert_pretrained = cfg.model.projection.get("mert_layer_weights_path", None)
    
    # Skip checkpoint loading if using Q-Former with pretrained paths (already loaded in model init)
    if projection_type == "qformer" and (qformer_pretrained or mert_pretrained):
        logger.info("Q-Former projection with pretrained weights - skipping checkpoint loading (already loaded)")
    elif freeze_projection or freeze_layer_weights:
        # Get checkpoint path from resume config
        checkpoint_path = cfg.training.resume.get("checkpoint_path", None)
        if checkpoint_path is None:
            raise ValueError(
                "freeze_projection or freeze_layer_weights is enabled but no checkpoint_path provided in training.resume "
                "(and no pretrained_path specified for Q-Former projection)"
            )
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
        
        logger.info("Loading pre-trained weights from checkpoint: %s", checkpoint_path)
        
        # Load and freeze projection weights
        if freeze_projection:
            projection_path = os.path.join(checkpoint_path, "audio_projection.bin")
            if not os.path.exists(projection_path):
                raise ValueError(
                    f"Projection weights not found at: {projection_path}"
                )
            
            logger.info("Loading and freezing projection weights from %s", projection_path)
            projection_state_dict = torch.load(
                projection_path, 
                map_location="cuda" if torch.cuda.is_available() else "cpu"
            )
            model.audio_projection.load_state_dict(projection_state_dict)
            
            # Freeze all projection parameters
            for param in model.audio_projection.parameters():
                param.requires_grad = False
            
            logger.info("Projection weights loaded and frozen")
        
        # Load and freeze MERT layer weights
        if freeze_layer_weights:
            mert_path = os.path.join(checkpoint_path, "mert_encoder.bin")
            if os.path.exists(mert_path):
                logger.info("Loading and freezing MERT layer_weights from %s", mert_path)
                mert_state_dict = torch.load(
                    mert_path,
                    map_location="cuda" if torch.cuda.is_available() else "cpu"
                )
                model.audio_encoder.load_state_dict(mert_state_dict)
                
                # Ensure layer_weights are frozen (should already be from encoder init)
                model.audio_encoder.layer_weights.requires_grad = False
                logger.info("MERT layer_weights loaded and frozen")
            else:
                logger.warning(
                    "freeze_layer_weights is True but no mert_encoder.bin found at %s. "
                    "Layer weights will use default initialization.",
                    mert_path
                )
    
    logger.info("Model and tokenizer initialized.")
    model.print_trainable_parameters()
    return model, tokenizer
