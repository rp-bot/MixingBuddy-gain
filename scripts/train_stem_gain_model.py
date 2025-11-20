"""
Training script for stem classification and gain regression model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import torch
import logging
from torch.utils.data import Subset
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction
import numpy as np

# Allowlist required globals for torch.load(weights_only=True) introduced in PyTorch 2.6
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

# Configure logging to ensure it goes to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure transformers Trainer also logs to stdout
transformers_logger = logging.getLogger("transformers.trainer")
transformers_logger.setLevel(logging.INFO)
transformers_logger.addHandler(logging.StreamHandler(sys.stdout))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stem_gain_dataset import StemGainDataset  # noqa: E402
from src.data.stem_gain_collator import StemGainDataCollator  # noqa: E402
from src.training.stem_gain_trainer import StemGainTrainer  # noqa: E402
from src.models.stem_gain_model import StemGainModel  # noqa: E402
from src.utils.model_utils import initialize_experiment_tracker, generate_run_name  # noqa: E402


class LoggingCallback(TrainerCallback):
    """Callback to log training metrics to stdout."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to stdout."""
        logger.info(f"Step {state.global_step}")


class StemGainCheckpointCallback(TrainerCallback):
    """Callback to save custom model components during checkpointing."""
    
    def __init__(self, model: StemGainModel):
        self.model = model
    
    def on_save(self, args, state, control, **kwargs):
        """Save custom components when checkpoint is saved."""
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        self._save_custom_components(checkpoint_dir)
    
    def _save_custom_components(self, checkpoint_dir: str):
        """Save custom model components."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save classification head
        torch.save(
            self.model.classification_head.state_dict(),
            f"{checkpoint_dir}/classification_head.bin",
        )
        
        # Save regression head
        torch.save(
            self.model.regression_head.state_dict(),
            f"{checkpoint_dir}/regression_head.bin",
        )
        
        # Save projection if used
        if self.model.use_projection and self.model.projection is not None:
            torch.save(
                self.model.projection.state_dict(),
                f"{checkpoint_dir}/projection.bin",
            )
        
        # Save encoder weights if trainable (MERT layer_weights)
        if hasattr(self.model.audio_encoder, "layer_weights"):
            torch.save(
                self.model.audio_encoder.state_dict(),
                f"{checkpoint_dir}/mert_encoder.bin",
            )
        elif not self.model.audio_encoder.frozen:
            # Encoder is trainable but not MERT (e.g., EnCodec)
            torch.save(
                self.model.audio_encoder.state_dict(),
                f"{checkpoint_dir}/encoder.bin",
            )
        
        logger.info(f"Saved custom components to {checkpoint_dir}/")


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute evaluation metrics for stem classification and gain regression.
    
    Args:
        eval_pred: EvalPrediction with:
            - predictions: tuple(classification_logits, gain_predictions)
            - label_ids: tuple(stem_labels, gain_labels)
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Unpack predictions and labels
    classification_logits, gain_predictions = predictions
    stem_labels, gain_labels = labels
    
    # Ensure numpy arrays
    classification_logits = np.asarray(classification_logits)
    gain_predictions = np.asarray(gain_predictions)
    stem_labels = np.asarray(stem_labels)
    gain_labels = np.asarray(gain_labels)
    
    # Stem classification accuracy
    stem_preds = classification_logits.argmax(axis=-1)
    stem_accuracy = (stem_preds == stem_labels).mean().item()
    
    # Gain regression metrics (in dB)
    gain_errors = gain_predictions - gain_labels
    mae = np.mean(np.abs(gain_errors)).item()
    rmse = np.sqrt(np.mean(gain_errors ** 2)).item()
    
    return {
        "stem_accuracy": stem_accuracy,
        "gain_mae_db": mae,
        "gain_rmse_db": rmse,
    }


@hydra.main(
    config_path="../configs",
    config_name="train_stem_gain_wav2vec",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main training function."""
    tracker = initialize_experiment_tracker(cfg, required=False)
    
    # Initialize model
    logger.info("Initializing model...")
    model = StemGainModel(
        encoder_config=cfg.model.get("encoder"),
        projection_config=cfg.model.get("projection"),
        num_classes=cfg.model.get("num_classes", 4),
        pooling_method=cfg.model.get("pooling_method", "mean"),
        head_config=cfg.model.get("head"),
    )
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = StemGainDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.get("limit"),
        random_seed=cfg.data.get("random_seed"),
        augmentation_config=cfg.data.get("augmentation"),
    )
    
    # Create validation dataset if paths are provided
    val_dataset = None
    if cfg.data.get("val_jsonl_path") and cfg.data.get("val_audio_root"):
        val_dataset = StemGainDataset(
            jsonl_path=cfg.data.val_jsonl_path,
            audio_root=cfg.data.val_audio_root,
            sample_rate=cfg.data.audio.sample_rate,
            limit=cfg.data.get("limit"),
            random_seed=cfg.data.get("random_seed"),
            # Do not use augmentation for validation set
            augmentation_config=None,
        )

    # Optionally cap validation set size for faster evaluation
    eval_cfg = cfg.training.get("evaluation")
    if val_dataset is not None and eval_cfg is not None:
        max_eval_samples = eval_cfg.get("max_eval_samples")
        if max_eval_samples is not None and len(val_dataset) > max_eval_samples:
            logger.info(
                "Limiting validation set from %d to %d samples for faster evaluation",
                len(val_dataset),
                max_eval_samples,
            )
            indices = list(range(max_eval_samples))
            val_dataset = Subset(val_dataset, indices)
    
    # Prepare training arguments
    training_args_dict = OmegaConf.to_container(
        cfg.training.training_args, resolve=True
    )
    training_args_dict["remove_unused_columns"] = False
    # Ensure Trainer keeps our label fields for metrics
    training_args_dict["label_names"] = ["stem_label", "gain_label"]
    
    if tracker:
        run_name = tracker._current_run_name
    else:
        # Use shared naming utility (now supports non-LoRA models like stem_gain)
        run_name = generate_run_name(cfg)
    
    base_output_dir_cfg = training_args_dict["output_dir"]
    base_output_dir_path = Path(base_output_dir_cfg)
    if not base_output_dir_path.is_absolute():
        base_output_dir_path = (PROJECT_ROOT / base_output_dir_path).resolve()
    training_args_dict["output_dir"] = str(base_output_dir_path / run_name)
    logger.info("Checkpoints will be saved to: %s", training_args_dict["output_dir"])
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Setup callbacks
    callbacks = []
    callbacks.append(LoggingCallback())  # Add logging callback first
    callbacks.append(StemGainCheckpointCallback(model))
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping.patience
            )
        )
    
    # Data collator
    data_collator = StemGainDataCollator(
        pad_value=cfg.data.get("pad_value", 0.0)
    )
    
    # Trainer
    trainer = StemGainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        classification_weight=cfg.training.get("classification_weight", 1.0),
        regression_weight=cfg.training.get("regression_weight", 0.1),
        compute_metrics=compute_metrics,
    )
    
    # Define output directory
    final_model_dir = training_args.output_dir
    
    # Handle resume from checkpoint
    # Check top-level resume config first, then fall back to training.resume
    resume_config = cfg.get("resume")
    if resume_config is None:
        resume_config = cfg.training.resume
    
    resume_from_checkpoint: str | None = None
    if resume_config.enabled:
        if resume_config.checkpoint_path is None:
            logger.warning("Resume enabled but checkpoint_path is None; starting from scratch")
        else:
            checkpoint_path = PROJECT_ROOT / resume_config.checkpoint_path
            weight_only = resume_config.get("weight_only", False)
            
            if checkpoint_path.exists():
                # Detect if this is a full HF checkpoint (has optimizer/scheduler/state)
                has_trainer_state = (checkpoint_path / "trainer_state.json").exists()
                has_optimizer = (checkpoint_path / "optimizer.pt").exists()
                
                if has_trainer_state and has_optimizer and not weight_only:
                    # Before resuming, ensure model architecture is compatible
                    architecture_compatible = True
                    model_bin = checkpoint_path / "pytorch_model.bin"
                    model_safe = checkpoint_path / "model.safetensors"
                    
                    if model_bin.exists() or model_safe.exists():
                        try:
                            if model_bin.exists():
                                ckpt_state = torch.load(model_bin, map_location="cpu", weights_only=True)
                            else:
                                try:
                                    from safetensors.torch import load_file as safe_load_file
                                    ckpt_state = safe_load_file(str(model_safe))
                                except Exception as se:
                                    logger.warning("Failed to read model.safetensors: %s", se)
                                    ckpt_state = None
                            
                            if ckpt_state is None:
                                architecture_compatible = False
                            else:
                                # Check head architectures for compatibility
                                current_cls_sd = model.classification_head.state_dict()
                                current_reg_sd = model.regression_head.state_dict()
                                
                                for k, v in ckpt_state.items():
                                    # Check classification head
                                    if k.startswith("classification_head."):
                                        key = k.replace("classification_head.", "")
                                        if key in current_cls_sd and current_cls_sd[key].shape != v.shape:
                                            architecture_compatible = False
                                            logger.info(
                                                "Classification head shape mismatch for %s: checkpoint %s vs current %s",
                                                key, v.shape, current_cls_sd[key].shape
                                            )
                                            break
                                    # Check regression head
                                    elif k.startswith("regression_head."):
                                        key = k.replace("regression_head.", "")
                                        if key in current_reg_sd and current_reg_sd[key].shape != v.shape:
                                            architecture_compatible = False
                                            logger.info(
                                                "Regression head shape mismatch for %s: checkpoint %s vs current %s",
                                                key, v.shape, current_reg_sd[key].shape
                                            )
                                            break
                        except Exception as arch_err:
                            logger.warning("Failed to validate checkpoint architecture: %s", arch_err)
                            architecture_compatible = False
                    else:
                        # No model weights file to validate against; avoid full resume
                        architecture_compatible = False
                    
                    if architecture_compatible:
                        resume_from_checkpoint = str(checkpoint_path)
                        logger.info(
                            "Resuming full training state from checkpoint: %s (epoch, step, LR scheduler)",
                            resume_from_checkpoint,
                        )
                    else:
                        logger.info(
                            "Detected architecture mismatch; performing weight-only warm start from: %s",
                            checkpoint_path,
                        )
                        # Fall through to weight-only warm start below
                        # (do not set resume_from_checkpoint)
                else:
                    if weight_only:
                        logger.info(
                            "Weight-only mode enabled; performing weight-only warm start from: %s",
                            checkpoint_path,
                        )
                    else:
                        logger.info(
                            "Checkpoint found but missing optimizer/scheduler; performing weight-only warm start from: %s",
                            checkpoint_path,
                        )
                    # Weight-only warm start (fall through to load custom components)
                
                # Load custom components for weight-only resume
                if resume_from_checkpoint is None:
                    _load_custom_components(model, checkpoint_path)
            else:
                logger.warning(
                    "Resume enabled but checkpoint not found at: %s; starting from scratch",
                    checkpoint_path,
                )
    else:
        logger.info("Resume disabled; starting from scratch")
    
    # Start training
    logger.info("Starting training...")
    try:
        def _attempt_train(resume_path: str | None) -> None:
            if resume_path is not None:
                # Pre-validate rng_state compatibility with PyTorch 2.6 weights_only semantics
                try:
                    rng_file = Path(resume_path) / "rng_state.pth"
                    if rng_file.exists():
                        try:
                            _ = torch.load(rng_file, weights_only=True)
                        except Exception:
                            try:
                                _ = torch.load(rng_file, weights_only=False)
                            except Exception:
                                logger.warning(
                                    "Incompatible rng_state.pth detected; deleting to proceed with resume."
                                )
                                try:
                                    rng_file.unlink()
                                except Exception as unlink_err:
                                    logger.warning(
                                        "Failed to delete rng_state.pth (%s); continuing without it may still fail.",
                                        unlink_err,
                                    )
                except Exception as rng_check_err:
                    logger.debug("RNG state pre-check failed: %s", rng_check_err)
                trainer.train(resume_from_checkpoint=resume_path)
            else:
                trainer.train()
        
        try:
            _attempt_train(resume_from_checkpoint)
        except RuntimeError as re:
            msg = str(re)
            if "size mismatch" in msg and ("classification_head" in msg or "regression_head" in msg) and resume_from_checkpoint is not None:
                logger.warning(
                    "Resume failed due to head shape mismatch; retrying with weight-only warm start and fresh trainer state."
                )
                resume_from_checkpoint = None
                _attempt_train(None)
            else:
                raise
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.info("Attempting to save current state...")
        try:
            trainer.save_model(final_model_dir)
            _save_custom_components(model, final_model_dir)
            logger.info("Partial model saved despite training failure.")
        except Exception as save_error:
            logger.error(f"Failed to save model: {save_error}")
        raise
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(final_model_dir)
    _save_custom_components(model, final_model_dir)
    
    logger.info("Model and custom components saved to %s", final_model_dir)
    
    if tracker:
        tracker.finish()


def _load_custom_components(model: StemGainModel, checkpoint_path: Path):
    """Load custom model components from checkpoint with shape mismatch handling."""
    # Load classification head
    cls_path = checkpoint_path / "classification_head.bin"
    if cls_path.exists():
        logger.info("Loading classification head from %s", cls_path)
        state_dict = torch.load(cls_path, map_location="cpu", weights_only=True)
        # Filter keys to only those that exist and have matching shapes
        current_sd = model.classification_head.state_dict()
        filtered_state_dict = {}
        skipped_mismatch = []
        for k, v in state_dict.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                filtered_state_dict[k] = v
            elif k in current_sd:
                skipped_mismatch.append(k)
        
        if skipped_mismatch:
            logger.info(
                "Skipping %d classification head params with shape mismatch (e.g., different head architecture): %s",
                len(skipped_mismatch),
                ", ".join(skipped_mismatch[:5]) + ("..." if len(skipped_mismatch) > 5 else ""),
            )
        
        missing_after = set(current_sd.keys()) - set(filtered_state_dict.keys())
        if missing_after:
            logger.info(
                "Classification head params not initialized from checkpoint: %d (will use default init)",
                len(missing_after),
            )
        
        model.classification_head.load_state_dict(filtered_state_dict, strict=False)
    
    # Load regression head
    reg_path = checkpoint_path / "regression_head.bin"
    if reg_path.exists():
        logger.info("Loading regression head from %s", reg_path)
        state_dict = torch.load(reg_path, map_location="cpu", weights_only=True)
        # Filter keys to only those that exist and have matching shapes
        current_sd = model.regression_head.state_dict()
        filtered_state_dict = {}
        skipped_mismatch = []
        for k, v in state_dict.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                filtered_state_dict[k] = v
            elif k in current_sd:
                skipped_mismatch.append(k)
        
        if skipped_mismatch:
            logger.info(
                "Skipping %d regression head params with shape mismatch (e.g., different head architecture): %s",
                len(skipped_mismatch),
                ", ".join(skipped_mismatch[:5]) + ("..." if len(skipped_mismatch) > 5 else ""),
            )
        
        missing_after = set(current_sd.keys()) - set(filtered_state_dict.keys())
        if missing_after:
            logger.info(
                "Regression head params not initialized from checkpoint: %d (will use default init)",
                len(missing_after),
            )
        
        model.regression_head.load_state_dict(filtered_state_dict, strict=False)
    
    # Load projection if used
    if model.use_projection:
        proj_path = checkpoint_path / "projection.bin"
        if proj_path.exists():
            logger.info("Loading projection from %s", proj_path)
            state_dict = torch.load(proj_path, map_location="cpu", weights_only=True)
            # Filter keys to only those that exist and have matching shapes
            current_sd = model.projection.state_dict()
            filtered_state_dict = {}
            skipped_mismatch = []
            for k, v in state_dict.items():
                if k in current_sd and current_sd[k].shape == v.shape:
                    filtered_state_dict[k] = v
                elif k in current_sd:
                    skipped_mismatch.append(k)
            
            if skipped_mismatch:
                logger.info(
                    "Skipping %d projection params with shape mismatch (e.g., different projection architecture): %s",
                    len(skipped_mismatch),
                    ", ".join(skipped_mismatch[:5]) + ("..." if len(skipped_mismatch) > 5 else ""),
                )
            
            missing_after = set(current_sd.keys()) - set(filtered_state_dict.keys())
            if missing_after:
                logger.info(
                    "Projection params not initialized from checkpoint: %d (will use default init)",
                    len(missing_after),
                )
            
            model.projection.load_state_dict(filtered_state_dict, strict=False)
    
    # Load encoder weights
    mert_path = checkpoint_path / "mert_encoder.bin"
    encoder_path = checkpoint_path / "encoder.bin"
    if mert_path.exists():
        logger.info("Loading MERT encoder weights from %s", mert_path)
        state_dict = torch.load(mert_path, map_location="cpu", weights_only=True)
        model.audio_encoder.load_state_dict(state_dict, strict=False)
    elif encoder_path.exists():
        logger.info("Loading encoder weights from %s", encoder_path)
        state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
        model.audio_encoder.load_state_dict(state_dict, strict=False)


def _save_custom_components(model: StemGainModel, output_dir: str):
    """Save custom model components."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification head
    torch.save(
        model.classification_head.state_dict(),
        f"{output_dir}/classification_head.bin",
    )
    
    # Save regression head
    torch.save(
        model.regression_head.state_dict(),
        f"{output_dir}/regression_head.bin",
    )
    
    # Save projection if used
    if model.use_projection and model.projection is not None:
        torch.save(
            model.projection.state_dict(),
            f"{output_dir}/projection.bin",
        )
    
    # Save encoder weights if trainable
    if hasattr(model.audio_encoder, "layer_weights"):
        torch.save(
            model.audio_encoder.state_dict(),
            f"{output_dir}/mert_encoder.bin",
        )
    elif not model.audio_encoder.frozen:
        torch.save(
            model.audio_encoder.state_dict(),
            f"{output_dir}/encoder.bin",
        )


if __name__ == "__main__":
    main()
