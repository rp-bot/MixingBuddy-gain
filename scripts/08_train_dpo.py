"""
DPO (Direct Preference Optimization) training script for multimodal model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import torch
import logging
from transformers import TrainingArguments, EarlyStoppingCallback

# Allowlist required globals for torch.load(weights_only=True)
try:
    import numpy as np  # noqa: F401
    safe_globals = []
    try:
        safe_globals.append(np.core.multiarray._reconstruct)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        safe_globals.append(np._core.multiarray._reconstruct)  # type: ignore[attr-defined]
    except Exception:
        pass
    if len(safe_globals) > 0:
        torch.serialization.add_safe_globals(safe_globals)
except Exception as e:
    print(f"Error adding safe globals: {e}")
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dpo_collator import DPOMultimodalDataCollator  # noqa: E402
from src.data.dpo_dataset import DPODataset  # noqa: E402
from src.training.dpo_trainer import DPOTrainer  # noqa: E402
from src.training.trainer import ExperimentTrackingCallback  # noqa: E402
from src.utils.model_utils import initialize_experiment_tracker  # noqa: E402
from src.training.callbacks import ProjectionDiagnosticCallback  # noqa: E402
from src.models.initialization import initialize_model_and_tokenizer  # noqa: E402
from src.utils.model_utils import IterableDatasetWrapper  # noqa: E402

logger = logging.getLogger(__name__)


def load_dpo_datasets(cfg: DictConfig, tokenizer):
    """Load DPO training and validation datasets."""
    logger.info(
        "Expected audio length: %d samples (%ss at %sHz)",
        int(cfg.data.chunk.sec * cfg.data.audio.sample_rate),
        cfg.data.chunk.sec,
        cfg.data.audio.sample_rate,
    )

    # Load full DPO training dataset
    full_train_pytorch_dataset = DPODataset(
        jsonl_path=cfg.data.train_dpo_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        system_message=cfg.data.system_message,
        limit=cfg.data.get("limit", None),
    )
    full_train_dataset = IterableDatasetWrapper(full_train_pytorch_dataset)

    # Load DPO test dataset
    test_pytorch_dataset = DPODataset(
        jsonl_path=cfg.data.test_dpo_jsonl_path,
        audio_root=cfg.data.test_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        system_message=cfg.data.system_message,
        limit=cfg.data.get("limit", None),
    )
    test_dataset = IterableDatasetWrapper(test_pytorch_dataset)

    # Split training into train/val
    train_val_split = full_train_dataset.train_test_split(
        test_size=0.2, seed=cfg.env.seed
    )
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    logger.info(
        "DPO Dataset sizes: Train=%d, Val=%d, Test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    return train_dataset, val_dataset, test_dataset


@hydra.main(
    config_path="../configs",
    config_name="25_train_dpo",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main DPO training function."""
    tracker = initialize_experiment_tracker(cfg, required=False)
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(cfg)
    
    # Load DPO datasets
    train_dataset, val_dataset, test_dataset = load_dpo_datasets(cfg, tokenizer)

    # Prepare training arguments
    training_args_dict = OmegaConf.to_container(
        cfg.training.training_args, resolve=True
    )
    training_args_dict["remove_unused_columns"] = False
    training_args_dict["disable_tqdm"] = True

    if tracker:
        run_name = tracker._current_run_name
    else:
        from src.utils.model_utils import generate_run_name
        run_name = generate_run_name(cfg)

    base_output_dir = training_args_dict["output_dir"]
    training_args_dict["output_dir"] = f"{base_output_dir}/{run_name}"
    logger.info("Checkpoints will be saved to: %s", training_args_dict["output_dir"])

    training_args = TrainingArguments(**training_args_dict)

    # Setup callbacks
    callbacks = []
    callbacks.append(ExperimentTrackingCallback(tracker, model))
    callbacks.append(ProjectionDiagnosticCallback(model))
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping.patience
            )
        )

    # Get audio encoder stride
    if hasattr(model.audio_encoder, "hop_length"):
        audio_encoder_stride = model.audio_encoder.hop_length
    elif hasattr(model.audio_encoder.model, "config") and hasattr(
        model.audio_encoder.model.config, "hop_length"
    ):
        audio_encoder_stride = model.audio_encoder.model.config.hop_length
    else:
        logger.warning("Could not determine audio encoder stride, using default")
        audio_encoder_stride = 320

    logger.info(f"Audio encoder stride: {audio_encoder_stride}")

    # Create DPO data collator
    data_collator = DPOMultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        audio_encoder_stride=audio_encoder_stride,
    )

    # Create reference model (frozen copy for DPO)
    logger.info("Creating reference model for DPO...")
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model created and frozen")

    # Initialize DPO trainer
    dpo_beta = cfg.training.dpo.get("beta", 0.1)
    dpo_loss_type = cfg.training.dpo.get("loss_type", "sigmoid")
    
    logger.info(f"DPO hyperparameters: beta={dpo_beta}, loss_type={dpo_loss_type}")

    # Only include eval_dataset if evaluation is enabled
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = val_dataset
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=dpo_beta,
        loss_type=dpo_loss_type,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Define output directory
    final_model_dir = training_args.output_dir

    # Handle checkpoint resuming
    # For DPO, we typically want to load model weights from SFT but start fresh optimizer state
    resume_from_checkpoint: str | None = None
    load_model_weights_only = False
    
    if cfg.training.resume.enabled:
        checkpoint_path = PROJECT_ROOT / cfg.training.resume.checkpoint_path
        if checkpoint_path.exists():
            # For DPO, we want to load model weights but not optimizer/scheduler state
            # because the training objective is different
            if cfg.training.resume.get("weight_only", True):
                load_model_weights_only = True
                logger.info(
                    "Loading model weights only from checkpoint (DPO requires fresh optimizer state): %s",
                    checkpoint_path,
                )
                # Load projection weights if they exist
                projection_path = checkpoint_path / "audio_projection.bin"
                if projection_path.exists():
                    logger.info("Loading projection weights from %s", projection_path)
                    state_dict = torch.load(projection_path, map_location="cpu")
                    current_sd = model.audio_projection.state_dict()
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if k in current_sd and current_sd[k].shape == v.shape
                    }
                    model.audio_projection.load_state_dict(filtered_state_dict, strict=False)
                
                # Load MERT encoder weights if they exist
                mert_path = checkpoint_path / "mert_encoder.bin"
                if mert_path.exists():
                    logger.info("Loading MERT encoder weights from %s", mert_path)
                    mert_state_dict = torch.load(mert_path, map_location="cpu")
                    model.audio_encoder.load_state_dict(mert_state_dict)
                
                # Load LoRA adapters if they exist
                adapter_path = checkpoint_path / "adapter_model.safetensors"
                if adapter_path.exists():
                    logger.info("Loading LoRA adapter weights from %s", adapter_path)
                    from safetensors.torch import load_file
                    
                    # Load the adapter weights
                    adapter_state_dict = load_file(str(adapter_path))
                    
                    # Model already has LoRA initialized, load weights into existing structure
                    model_state = model.llm.state_dict()
                    filtered_state = {}
                    
                    # Log sample keys for debugging
                    adapter_sample = list(adapter_state_dict.keys())[0]
                    model_sample = list(model_state.keys())[0]
                    logger.info(f"Adapter key sample: {adapter_sample}")
                    logger.info(f"Model key sample: {model_sample}")
                    
                    for key, value in adapter_state_dict.items():
                        # Try different key transformations to match model structure
                        candidate_keys = [
                            key,  # Try original key first
                            key.replace("base_model.model.", ""),  # Remove one level of nesting
                            # Add .default adapter name if missing (PEFT saves without it, but loads with it)
                            key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
                            # Try both transformations combined
                            key.replace("base_model.model.", "").replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"),
                        ]
                        
                        for candidate_key in candidate_keys:
                            if candidate_key in model_state and model_state[candidate_key].shape == value.shape:
                                filtered_state[candidate_key] = value
                                break
                    
                    missing, unexpected = model.llm.load_state_dict(filtered_state, strict=False)
                    logger.info(f"Loaded {len(filtered_state)} LoRA parameters")
                    logger.info(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            else:
                # Full resume (not recommended for DPO from SFT checkpoint)
                resume_from_checkpoint = str(checkpoint_path)
                logger.warning(
                    "Full resume enabled - optimizer state may not match. "
                    "Consider using weight_only=true for DPO training."
                )
        else:
            logger.warning(
                "Resume enabled but checkpoint not found at: %s; starting from scratch",
                checkpoint_path,
            )
    else:
        logger.info("Resume disabled; starting from scratch")

    # Start training
    logger.info("Starting DPO training...")
    try:
        # If we loaded weights only, don't pass resume_from_checkpoint
        if load_model_weights_only:
            trainer.train(resume_from_checkpoint=None)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("DPO training finished successfully.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.info("Attempting to save current state...")
        try:
            trainer.save_model(final_model_dir)
            torch.save(
                model.audio_projection.state_dict(),
                f"{final_model_dir}/audio_projection.bin",
            )
            if hasattr(model.audio_encoder, "layer_weights"):
                torch.save(
                    model.audio_encoder.state_dict(),
                    f"{final_model_dir}/mert_encoder.bin",
                )
            logger.info("Partial model saved despite training failure.")
        except Exception as save_error:
            logger.error(f"Failed to save model: {save_error}")
        raise

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(final_model_dir)

    logger.info("Saving custom model components...")
    torch.save(
        model.audio_projection.state_dict(),
        f"{final_model_dir}/audio_projection.bin",
    )

    if hasattr(model.audio_encoder, "layer_weights"):
        logger.info("Saving MERT encoder weights...")
        torch.save(
            model.audio_encoder.state_dict(),
            f"{final_model_dir}/mert_encoder.bin",
        )

    target_modules = cfg.model.lora.get("target_modules", [])
    if len(target_modules) > 0:
        logger.info("Saving PEFT adapter files...")
        model.llm.save_pretrained(final_model_dir)
    else:
        logger.info("Skipping PEFT save (projection-only training)")

    logger.info("Model and custom components saved to %s", final_model_dir)

    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()

