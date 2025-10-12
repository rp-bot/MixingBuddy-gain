"""
Model evaluation script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import torch
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import LoRATrainer  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    find_latest_checkpoint,
    load_dataset,
)


def load_trained_model(cfg: DictConfig):
    """Load a trained model for evaluation."""
    from transformers import AutoTokenizer
    from peft import PeftModel

    print("Loading trained model for evaluation...")

    # Set memory management environment variable early
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Create LoRA configuration
    lora_config = create_lora_config(cfg)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if cfg.model.use_qlora:
        llm = initialize_qlora_model(cfg, lora_config, tokenizer)
    else:
        llm = initialize_lora_model(cfg, lora_config, tokenizer)

    # Determine checkpoint path
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint found when checkpoint_path is set to 'latest'"
            )

    # Load the trained LoRA weights if checkpoint path is provided
    if checkpoint_path:
        print(f"Loading LoRA weights from {checkpoint_path}")
        llm = PeftModel.from_pretrained(llm, checkpoint_path)

    # Initialize the multimodal model
    model = ModularMultimodalModel(
        model_name=cfg.model.model_name,
        use_qlora=cfg.model.use_qlora,
        lora_config=lora_config,
        llm=llm,
        tokenizer=tokenizer,
    )

    # Load audio projection weights if available
    if checkpoint_path and cfg.evaluation.model_loading.load_audio_projection:
        projection_path = f"{checkpoint_path}/audio_projection.bin"
        if Path(projection_path).exists():
            print(f"Loading audio projection weights from {projection_path}")
            map_location = cfg.evaluation.model_loading.map_location
            if map_location == "auto":
                map_location = "cuda" if torch.cuda.is_available() else "cpu"

            model.audio_projection.load_state_dict(
                torch.load(projection_path, map_location=map_location),
                strict=cfg.evaluation.model_loading.strict_loading,
            )
        else:
            print(
                "Warning: Audio projection weights not found, using random initialization"
            )

    print("Model loaded for evaluation.")
    model.print_trainable_parameters()

    # Clean up memory after model loading
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return model


def load_test_dataset(cfg: DictConfig, model):
    """Load test dataset for evaluation."""
    # Use evaluation-specific limit if set, otherwise use data limit
    limit = (
        cfg.evaluation.max_samples
        if cfg.evaluation.max_samples
        else cfg.data.get("limit")
    )

    return load_dataset(cfg, model, "test", limit)


def evaluate_model_directly(model, test_dataset, cfg):
    """Evaluate model directly without HuggingFace Trainer to save memory."""
    import math
    from torch.utils.data import DataLoader
    from src.data.collator import MultimodalDataCollator

    print("Setting up direct evaluation...")

    # Set model to evaluation mode
    model.eval()

    # Create data collator
    data_collator = MultimodalDataCollator(
        tokenizer=model.tokenizer,
        pad_to_multiple_of=8,
    )

    # Create dataloader with small batch size
    eval_batch_size = cfg.evaluation.batch_size
    dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        pin_memory=False,
        num_workers=0,
    )

    total_loss = 0.0
    total_samples = 0

    print(
        f"Evaluating on {len(test_dataset)} samples with batch size {eval_batch_size}"
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                if torch.cuda.is_available():
                    # Get device from model parameters
                    device = next(model.parameters()).device
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                total_loss += loss.item()
                total_samples += batch["input_ids"].size(0)

                # Memory cleanup after each batch
                del outputs, loss
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    # Calculate metrics
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

    results = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_samples": total_samples,
    }

    print(f"Direct evaluation completed: {results}")
    return results


def run_evaluation(cfg: DictConfig):
    """Run evaluation on the test dataset."""
    print("Starting evaluation...")

    # Initialize model
    model = load_trained_model(cfg)

    # No experiment tracker needed for evaluation

    # Load test dataset
    test_dataset = load_test_dataset(cfg, model)

    # Skip trainer initialization for memory efficiency

    # Memory cleanup before evaluation if configured
    if cfg.evaluation.memory.cleanup_before_eval:
        print("Cleaning up memory before evaluation...")
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Set memory management environment variable
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"GPU memory before evaluation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Check if we have enough memory for evaluation
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_memory = torch.cuda.memory_allocated() / 1e9
    free_memory = total_memory - allocated_memory

    print(f"Total GPU memory: {total_memory:.2f} GB")
    print(f"Free GPU memory: {free_memory:.2f} GB")

    if free_memory < 2.0:  # Need at least 2GB for evaluation
        print(
            "Warning: Low GPU memory available. Consider reducing batch size or using CPU for some operations."
        )

    try:
        # Run direct evaluation without trainer
        print("Running direct evaluation on test dataset...")
        test_results = evaluate_model_directly(model, test_dataset, cfg)
        print(f"Evaluation results: {test_results}")

        # Save predictions if configured
        if cfg.evaluation.save_predictions:
            save_predictions(cfg, model, test_dataset)

        return test_results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Evaluation failed.")
        return {"eval_loss": "N/A", "eval_perplexity": "N/A"}


def save_predictions(cfg: DictConfig, model, test_dataset):
    """Save model predictions to file."""
    print("Saving model predictions...")

    predictions_dir = Path(cfg.evaluation.predictions_output_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # This is a placeholder for saving predictions
    # You can implement specific prediction saving logic here
    predictions_file = predictions_dir / "predictions.json"

    # For now, just create an empty file as placeholder
    with open(predictions_file, "w") as f:
        f.write("{}")

    print(f"Predictions saved to {predictions_file}")


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    """
    Main evaluation function.
    """
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Print configuration
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(f"Max samples: {cfg.evaluation.max_samples}")
    print(f"Save predictions: {cfg.evaluation.save_predictions}")
    print("=" * 50)

    # Run evaluation
    results = run_evaluation(cfg)

    print("=" * 50)
    print("EVALUATION COMPLETE")
    print(f"Results: {results}")
    print("=" * 50)


if __name__ == "__main__":
    main()
