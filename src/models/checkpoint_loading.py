import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from peft import PeftModel, prepare_model_for_kbit_training
from safetensors import safe_open
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.model_utils import find_latest_checkpoint, initialize_tokenizer


logger = logging.getLogger(__name__)


def load_trained_model(cfg: DictConfig) -> ModularMultimodalModel:
    """Load a trained model for evaluation."""
    logger.info("Loading trained model for generation...")

    checkpoint_path: Any = cfg.checkpoint_path
    if checkpoint_path == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint found when checkpoint_path is set to 'latest'"
            )
    
    # Load tokenizer from checkpoint if available, otherwise use base tokenizer
    # This ensures vocabulary size matches what was used during training
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
        logger.info("Loading tokenizer from checkpoint: %s", checkpoint_path)
        tokenizer = initialize_tokenizer(checkpoint_path)
    else:
        logger.info("Loading base tokenizer from: %s", cfg.model.model_name)
        tokenizer = initialize_tokenizer(cfg.model.model_name)

    logger.info("Tokenizer vocabulary size: %d", len(tokenizer))
    
    # Check if checkpoint has LoRA adapters with different vocab size
    # This handles cases where embeddings were resized during training
    # but the saved tokenizer doesn't reflect that
    expected_vocab_size = len(tokenizer)
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        # Try to detect vocab size from adapter weights
        adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            with safe_open(adapter_file, framework="pt") as f:
                # Check lm_head size to determine expected vocab size
                for key in f.keys():
                    if "lm_head" in key and "base_layer.weight" in key:
                        tensor_shape = f.get_tensor(key).shape
                        detected_vocab_size = tensor_shape[0]
                        if detected_vocab_size != len(tokenizer):
                            logger.warning(
                                "Detected vocab size mismatch: tokenizer has %d tokens, "
                                "but checkpoint LoRA weights expect %d tokens. "
                                "Will resize to match checkpoint.",
                                len(tokenizer), detected_vocab_size
                            )
                            expected_vocab_size = detected_vocab_size
                        break
    
    if cfg.model.use_qlora:
        logger.info("Loading base model with QLoRA quantization...")
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
        
        # Resize embeddings BEFORE preparing for kbit training
        logger.info("Resizing token embeddings to %d", expected_vocab_size)
        llm.resize_token_embeddings(expected_vocab_size)
        logger.info("Model vocabulary size after resize: %d", llm.get_input_embeddings().weight.shape[0])
        
        llm = prepare_model_for_kbit_training(llm)
    else:
        logger.info("Loading base model without quantization...")
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name,
            torch_dtype="auto",
        )
        llm.resize_token_embeddings(expected_vocab_size)

    has_lora_adapters = (
        os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
        if checkpoint_path
        else False
    )

    if checkpoint_path and has_lora_adapters:
        logger.info("Loading LoRA weights from %s", checkpoint_path)
        llm = PeftModel.from_pretrained(llm, checkpoint_path)
    elif checkpoint_path:
        logger.info(
            "No LoRA adapters found - projection-only model; using base LLM without adapters"
        )
        for param in llm.parameters():
            param.requires_grad = False

    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),
        projection_config=cfg.model.get("projection"),
    )

    use_random_projection = cfg.evaluation.get("use_random_projection", False)
    if use_random_projection:
        logger.warning("Using RANDOM audio projection weights for ablation study")
    else:
        projection_path = f"{checkpoint_path}/audio_projection.bin"
        logger.info("Loading audio projection weights from %s", projection_path)
        # Get model_loading config with defaults
        model_loading = cfg.evaluation.get("model_loading", {})
        map_location = model_loading.get("map_location", "auto")
        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(projection_path, map_location=map_location)
        strict_loading = model_loading.get("strict_loading", True)
        model.audio_projection.load_state_dict(
            state_dict,
            strict=strict_loading,
        )

    # Load MERT encoder weights if available (contains trainable layer_weights)
    mert_path = f"{checkpoint_path}/mert_encoder.bin"
    if os.path.exists(mert_path):
        logger.info("Loading MERT encoder weights from %s", mert_path)
        mert_state_dict = torch.load(mert_path, map_location=map_location)
        model.audio_encoder.load_state_dict(mert_state_dict)
    else:
        logger.info("No MERT encoder weights found, using default initialization")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded for generation.")
    model.print_trainable_parameters()
    return model
