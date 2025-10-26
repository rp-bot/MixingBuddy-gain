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
    config_name="10_train_projection_only",
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
    training_args_dict["disable_tqdm"] = False

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

    logger.info("Starting training... (optimizer diagnostic at step 0)")
    trainer.train()
    logger.info("Training finished.")

    logger.info("Saving final model...")
    final_model_dir = training_args.output_dir
    trainer.save_model(final_model_dir)

    logger.info("Saving custom model components (audio projection)...")
    torch.save(
        model.audio_projection.state_dict(),
        f"{final_model_dir}/audio_projection.bin",
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
