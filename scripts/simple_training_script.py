"""
Simple training script for instrument classification.
Uses MERT encoder, Qwen LLM, and MLP projection without config files.
"""

import sys
from pathlib import Path
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.simple_dataset import SimpleInstrumentDataset
from src.data.collator import MultimodalDataCollator
from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.model_utils import IterableDatasetWrapper
from src.training.trainer import ExperimentTrackingCallback
from src.training.callbacks import ProjectionDiagnosticCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""

    # Configuration (hardcoded for simplicity)
    model_name = "Qwen/Qwen2-7B-Instruct"
    audio_root = (
        PROJECT_ROOT / "data"
    )  # This will be used as base, but paths in JSONL already include "data/"
    train_jsonl = PROJECT_ROOT / "data" / "inst_classify_train.jsonl"
    test_jsonl = PROJECT_ROOT / "data" / "inst_classify_test.jsonl"
    sample_rate = 32000

    # Training arguments
    output_dir = PROJECT_ROOT / "outputs" / "simple_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=60,  # Train much longer since we're still improving
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Match config
        lr_scheduler_type="constant",  # Constant LR to eliminate scheduler issues
        learning_rate=5e-6,  # Higher LR for more aggressive learning
        weight_decay=0.001,  # Lower weight decay for more aggressive learning
        warmup_ratio=0.01,  # Much shorter warmup to get to full LR faster
        max_grad_norm=0.8,  # Higher gradient clipping for more aggressive learning
        logging_steps=5,  # Match config
        eval_steps=100,  # Match config
        save_steps=100,  # Match config
        logging_first_step=True,
        logging_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,  # Match config
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        disable_tqdm=False,
        # Don't use fp16/bf16 training with QLoRA - quantization handles precision
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading LLM with QLoRA quantization...")
    from transformers import BitsAndBytesConfig

    # Use QLoRA quantization with fp16 for compatibility
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),  # Use fp16 for compatibility
        bnb_4bit_use_double_quant=True,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Explicitly use fp16 for consistency
        quantization_config=quantization_config,
    )

    # Freeze LLM parameters
    for param in llm.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing for memory efficiency (match existing code)
    if hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    logger.info("Initializing multimodal model...")
    # Configure MERT encoder
    encoder_config = {
        "model_name": "m-a-p/MERT-v1-330M",
        "freeze": True,
        "device": llm.device,
        "input_sample_rate": sample_rate,
    }

    # Configure MLP projection to match config
    projection_config = {
        "type": "mlp",
        "hidden_dims": [2048, 4096, 4096, 2048],  # Match config
        "activation": "relu",
        "dropout": 0.1,
        "use_layer_norm": True,
        "use_residual": False,
        "use_auxiliary_loss": True,  # Match config
        "auxiliary_loss_weight": 0.05,  # Match config
    }

    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=encoder_config,
        projection_config=projection_config,
    )

    logger.info("Model initialized:")
    model.print_trainable_parameters()

    logger.info("Loading datasets...")
    # Load training data
    train_pytorch_dataset = SimpleInstrumentDataset(
        jsonl_path=train_jsonl,
        audio_root=audio_root,
        sample_rate=sample_rate,
        limit=1000,  # Limit for quick training
    )
    train_dataset = IterableDatasetWrapper(train_pytorch_dataset)

    # Load test data
    test_pytorch_dataset = SimpleInstrumentDataset(
        jsonl_path=test_jsonl,
        audio_root=audio_root,
        sample_rate=sample_rate,
        limit=200,  # Limit for quick evaluation
    )
    test_dataset = IterableDatasetWrapper(test_pytorch_dataset)

    # Split training data into train/val using the wrapper's method
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_subset = train_val_split["train"]
    val_subset = train_val_split["test"]

    logger.info(
        f"Dataset sizes: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}"
    )

    # Get audio encoder stride for data collator
    if hasattr(model.audio_encoder, "hop_length"):
        audio_encoder_stride = model.audio_encoder.hop_length
    else:
        audio_encoder_stride = 320  # Default stride

    logger.info(f"Audio encoder stride: {audio_encoder_stride}")

    # Create data collator
    data_collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        audio_encoder_stride=audio_encoder_stride,
    )

    # Create callbacks (same as main training script)
    callbacks = [
        ExperimentTrackingCallback(
            experiment_tracker=None, model=model
        ),  # Save audio projection at each checkpoint
        ProjectionDiagnosticCallback(model),  # Diagnostic logging for projection
        # EarlyStoppingCallback(early_stopping_patience=3),  # Disabled since we want to train longer
    ]

    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Define output directory before training
    final_model_dir = training_args.output_dir

    # Load weights from checkpoint but don't resume training state
    checkpoint_path = PROJECT_ROOT / "outputs" / "simple_training" / "checkpoint-100"
    if checkpoint_path.exists():
        logger.info(f"Loading weights from checkpoint: {checkpoint_path}")

        # Load the trained weights from the checkpoint
        projection_path = checkpoint_path / "audio_projection.bin"
        if projection_path.exists():
            logger.info(f"Loading projection weights from {projection_path}")
            state_dict = torch.load(projection_path, map_location="cpu")
            model.audio_projection.load_state_dict(state_dict)

        # Load MERT encoder weights if available
        mert_path = checkpoint_path / "mert_encoder.bin"
        if mert_path.exists():
            logger.info(f"Loading MERT encoder weights from {mert_path}")
            mert_state_dict = torch.load(mert_path, map_location="cpu")
            model.audio_encoder.load_state_dict(mert_state_dict)

        logger.info(
            "Starting fresh training with new hyperparameters (not resuming training state)"
        )
    else:
        logger.info("No checkpoint found, starting training from scratch")

    logger.info("Starting training...")
    try:
        trainer.train()  # Don't resume - use new hyperparameters
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
            torch.save(
                model.audio_encoder.state_dict(),
                f"{final_model_dir}/mert_encoder.bin",
            )
            logger.info("Partial model saved despite training failure.")
        except Exception as save_error:
            logger.error(f"Failed to save model: {save_error}")
        raise

    logger.info("Saving model...")
    trainer.save_model(final_model_dir)

    # Save custom model components
    logger.info("Saving audio projection...")
    torch.save(
        model.audio_projection.state_dict(),
        f"{final_model_dir}/audio_projection.bin",
    )

    # Save MERT encoder weights (including the 25 trainable layer weights)
    logger.info("Saving MERT encoder weights...")
    torch.save(
        model.audio_encoder.state_dict(),
        f"{final_model_dir}/mert_encoder.bin",
    )

    logger.info(f"Model saved to {final_model_dir}")

    # Quick evaluation on test set
    logger.info("Running evaluation on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    logger.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
