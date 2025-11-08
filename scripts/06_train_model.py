"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import torch
import logging
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

# Allowlist required globals for torch.load(weights_only=True) introduced in PyTorch 2.6
# This is safe here because checkpoints are produced by our own training pipeline.
try:
    import numpy as np  # noqa: F401
    safe_globals = []
    # Common path in NumPy
    try:
        safe_globals.append(np.core.multiarray._reconstruct)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Some environments reference it via np._core
    try:
        safe_globals.append(np._core.multiarray._reconstruct)  # type: ignore[attr-defined]
    except Exception:
        pass
    if len(safe_globals) > 0:
        torch.serialization.add_safe_globals(safe_globals)
except Exception as e:
    # If anything goes wrong, fall back silently; Trainer may still work depending on PyTorch version
    print(f"Error adding safe globals: {e}")
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collator import MultimodalDataCollator  # noqa: E402
from src.training.trainer import ExperimentTrackingCallback  # noqa: E402
from src.utils.model_utils import initialize_experiment_tracker  # noqa: E402
from src.training.callbacks import ProjectionDiagnosticCallback  # noqa: E402
from src.models.initialization import initialize_model_and_tokenizer  # noqa: E402
from src.data.loading import load_datasets  # noqa: E402


logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="12_train_mert_musdb_expanded_deeper",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main training function."""
    tracker = initialize_experiment_tracker(cfg, required=False)
    model, tokenizer = initialize_model_and_tokenizer(cfg)

    train_dataset, val_dataset, test_dataset = load_datasets(cfg, tokenizer)

    training_args_dict = OmegaConf.to_container(
        cfg.training.training_args, resolve=True
    )
    training_args_dict["remove_unused_columns"] = False
    training_args_dict["label_names"] = ["labels"]
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

    callbacks = []

    callbacks.append(ExperimentTrackingCallback(tracker, model))
    callbacks.append(ProjectionDiagnosticCallback(model))
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping.patience
            )
        )

    # Get hop_length from encoder (handles both EnCodec and MERT)
    if hasattr(model.audio_encoder, "hop_length"):
        # MERT and other encoders with hop_length property
        audio_encoder_stride = model.audio_encoder.hop_length
    elif hasattr(model.audio_encoder.model, "config") and hasattr(
        model.audio_encoder.model.config, "hop_length"
    ):
        # EnCodec-style encoders
        audio_encoder_stride = model.audio_encoder.model.config.hop_length
    else:
        # Fallback: try to infer from sample rate and output dimension
        logger.warning("Could not determine audio encoder stride, using default")
        audio_encoder_stride = 320  # Default stride

    logger.info(f"Audio encoder stride: {audio_encoder_stride}")

    data_collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        audio_encoder_stride=audio_encoder_stride,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Define output directory before training
    final_model_dir = training_args.output_dir

    # Resolve resume checkpoint if enabled
    resume_from_checkpoint: str | None = None
    if cfg.training.resume.enabled:
        checkpoint_path = PROJECT_ROOT / cfg.training.resume.checkpoint_path
        if checkpoint_path.exists():
            # Detect if this is a full HF checkpoint (has optimizer/scheduler/state)
            has_trainer_state = (checkpoint_path / "trainer_state.json").exists()
            has_optimizer = (checkpoint_path / "optimizer.pt").exists()
            if has_trainer_state and has_optimizer:
                # Before resuming, ensure model architecture is compatible. If projection shapes differ,
                # skip full resume and do a weight-only warm start instead.
                model_bin = checkpoint_path / "pytorch_model.bin"
                model_safe = checkpoint_path / "model.safetensors"
                architecture_compatible = True
                if model_bin.exists() or model_safe.exists():
                    try:
                        if model_bin.exists():
                            ckpt_state = torch.load(model_bin, map_location="cpu")
                        else:
                            try:
                                from safetensors.torch import load_file as safe_load_file  # type: ignore
                                ckpt_state = safe_load_file(str(model_safe))
                            except Exception as se:
                                logger.warning("Failed to read model.safetensors: %s", se)
                                ckpt_state = None
                        if ckpt_state is None:
                            architecture_compatible = False
                        else:
                            current_sd = model.audio_projection.state_dict()
                            for k, v in ckpt_state.items():
                                if not k.startswith("audio_projection."):
                                    continue
                                if k in current_sd and current_sd[k].shape != v.shape:
                                    architecture_compatible = False
                                    break
                    except Exception as arch_err:
                        logger.warning("Failed to validate checkpoint architecture: %s", arch_err)
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
                        "Detected projection architecture mismatch; performing weight-only warm start from: %s",
                        checkpoint_path,
                    )
                    # Fall through to weight-only warm start below
                    # (do not set resume_from_checkpoint)
            else:
                logger.info(
                    "Checkpoint found but missing optimizer/scheduler; performing weight-only warm start from: %s",
                    checkpoint_path,
                )
                # Weight-only warm start
                projection_path = checkpoint_path / "audio_projection.bin"
                if projection_path.exists():
                    logger.info("Loading projection weights from %s", projection_path)
                    state_dict = torch.load(projection_path, map_location="cpu")
                    # Filter keys to only those that exist and have matching shapes to allow
                    # architectural changes like deeper MLPs to warm start safely
                    current_sd = model.audio_projection.state_dict()
                    filtered_state_dict = {}
                    skipped_mismatch = []
                    for k, v in state_dict.items():
                        if k in current_sd and current_sd[k].shape == v.shape:
                            filtered_state_dict[k] = v
                        elif k in current_sd:
                            skipped_mismatch.append(k)
                    if skipped_mismatch:
                        logger.info(
                            "Skipping %d projection params with shape mismatch (e.g., deeper MLP): %s",
                            len(skipped_mismatch),
                            ", ".join(skipped_mismatch[:5]) + ("..." if len(skipped_mismatch) > 5 else ""),
                        )
                    missing_after = set(current_sd.keys()) - set(filtered_state_dict.keys())
                    if missing_after:
                        logger.info(
                            "Projection params not initialized from checkpoint: %d (will use default init)",
                            len(missing_after),
                        )
                    model.audio_projection.load_state_dict(filtered_state_dict, strict=False)
                mert_path = checkpoint_path / "mert_encoder.bin"
                if mert_path.exists():
                    logger.info("Loading MERT encoder weights from %s", mert_path)
                    mert_state_dict = torch.load(mert_path, map_location="cpu")
                    model.audio_encoder.load_state_dict(mert_state_dict)
        else:
            logger.warning(
                "Resume enabled but checkpoint not found at: %s; starting from scratch",
                checkpoint_path,
            )
    else:
        logger.info("Resume disabled; starting from scratch")

    logger.info("Starting training... (optimizer diagnostic at step 0)")
    try:
        def _attempt_train(resume_path: str | None) -> None:
            if resume_path is not None:
                # Pre-validate rng_state compatibility with PyTorch 2.6 weights_only semantics
                try:
                    rng_file = Path(resume_path) / "rng_state.pth"
                    if rng_file.exists():
                        try:
                            _ = torch.load(rng_file)
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
            if "size mismatch" in msg and "audio_projection" in msg and resume_from_checkpoint is not None:
                logger.warning(
                    "Resume failed due to projection shape mismatch; retrying with weight-only warm start and fresh trainer state."
                )
                resume_from_checkpoint = None
                _attempt_train(None)
            else:
                raise
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.info("Attempting to save current state...")
        # Try to save whatever we have
        try:
            trainer.save_model(final_model_dir)
            torch.save(
                model.audio_projection.state_dict(),
                f"{final_model_dir}/audio_projection.bin",
            )
            # Save MERT encoder weights if using MERT
            if hasattr(model.audio_encoder, "layer_weights"):
                torch.save(
                    model.audio_encoder.state_dict(),
                    f"{final_model_dir}/mert_encoder.bin",
                )
            logger.info("Partial model saved despite training failure.")
        except Exception as save_error:
            logger.error(f"Failed to save model: {save_error}")
        raise

    logger.info("Saving final model...")
    trainer.save_model(final_model_dir)

    logger.info("Saving custom model components (audio projection)...")
    torch.save(
        model.audio_projection.state_dict(),
        f"{final_model_dir}/audio_projection.bin",
    )

    # Save MERT encoder weights if using MERT (contains trainable layer_weights)
    if hasattr(model.audio_encoder, "layer_weights"):
        logger.info("Saving MERT encoder weights (trainable layer weights)...")
        torch.save(
            model.audio_encoder.state_dict(),
            f"{final_model_dir}/mert_encoder.bin",
        )

    target_modules = cfg.model.lora.get("target_modules", [])
    if len(target_modules) > 0:
        logger.info("Saving PEFT adapter files to final model directory...")
        model.llm.save_pretrained(final_model_dir)
    else:
        logger.info(
            "Skipping PEFT save (projection-only training - no adapters to save)"
        )

    logger.info("Model and custom components saved to %s", final_model_dir)

    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()
