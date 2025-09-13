#!/usr/bin/env python3
"""
Evaluation script for LoRA fine-tuned models.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import DataProcessor
from evaluation.metrics import compute_automatic_mixing_metrics
from models.lora_model import LoRAModel

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model evaluation...")

    # Print configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.env.seed)

    # Set device
    if config.env.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.env.device

    logger.info(f"Using device: {device}")

    try:
        # Load model from checkpoint
        checkpoint_path = config.get(
            "checkpoint_path", "outputs/checkpoints/final_model"
        )
        logger.info(f"Loading model from: {checkpoint_path}")

        model = LoRAModel(config.model)
        model.load_model()
        model.load_tokenizer()
        model.setup_lora()

        if Path(checkpoint_path).exists():
            model.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, using base model"
            )

        # Load test dataset
        logger.info("Loading test dataset...")
        data_processor = DataProcessor(config.data)
        test_dataset = data_processor.load_dataset("test")
        test_dataloader = data_processor.create_dataloader(test_dataset, "test")

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, test_dataloader, config)

        # Print results
        logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save results
        output_dir = Path(config.paths.output_dir) / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        import json

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Results saved to {output_dir / 'metrics.json'}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def evaluate_model(
    model: LoRAModel, dataloader, config: DictConfig
) -> Dict[str, float]:
    """Evaluate the model on the test dataset."""
    model.get_model().eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(model.get_model().device)
            attention_mask = batch["attention_mask"].to(model.get_model().device)
            labels = batch["labels"].to(model.get_model().device)

            # Forward pass
            outputs = model.get_model()(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_samples += input_ids.size(0)

            # Generate predictions
            predictions = model.generate(
                input_ids=input_ids,
                max_new_tokens=config.evaluation.max_new_tokens,
                temperature=config.evaluation.temperature,
                top_p=config.evaluation.top_p,
                top_k=config.evaluation.top_k,
                do_sample=config.evaluation.do_sample,
                pad_token_id=model.get_tokenizer().pad_token_id,
                eos_token_id=model.get_tokenizer().eos_token_id,
            )

            # Decode predictions and labels
            for i in range(predictions.size(0)):
                pred_text = model.get_tokenizer().decode(
                    predictions[i][input_ids.size(1) :], skip_special_tokens=True
                )
                label_text = model.get_tokenizer().decode(
                    labels[i][input_ids.size(1) :], skip_special_tokens=True
                )

                all_predictions.append(pred_text)
                all_labels.append(label_text)

    # Compute metrics
    avg_loss = total_loss / num_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Compute automatic mixing specific metrics
    mixing_metrics = compute_automatic_mixing_metrics(all_predictions, all_labels)

    # Combine all metrics
    metrics = {"loss": avg_loss, "perplexity": perplexity, **mixing_metrics}

    return metrics


if __name__ == "__main__":
    main()
