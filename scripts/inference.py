#!/usr/bin/env python3
"""
Inference script for LoRA fine-tuned models.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.lora_model import LoRAModel

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main inference function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model inference...")

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

        # Example prompts for automatic mixing
        prompts = [
            "Analyze the mixing parameters for this track:",
            "What EQ adjustments should I make for a vocal track?",
            "How should I compress a drum track?",
            "What reverb settings work best for acoustic guitar?",
            "How do I balance the bass and kick drum?",
        ]

        # Generate responses
        logger.info("Generating responses...")
        results = []

        for prompt in prompts:
            logger.info(f"Processing prompt: {prompt}")

            # Tokenize input
            inputs = model.get_tokenizer()(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            inputs = {k: v.to(model.get_model().device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=config.inference.max_new_tokens,
                    temperature=config.inference.temperature,
                    top_p=config.inference.top_p,
                    top_k=config.inference.top_k,
                    do_sample=config.inference.do_sample,
                    pad_token_id=model.get_tokenizer().pad_token_id,
                    eos_token_id=model.get_tokenizer().eos_token_id,
                )

            # Decode response
            response = model.get_tokenizer().decode(
                outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
            )

            result = {
                "prompt": prompt,
                "response": response,
                "full_text": model.get_tokenizer().decode(
                    outputs[0], skip_special_tokens=True
                ),
            }
            results.append(result)

            logger.info(f"Response: {response}")
            logger.info("-" * 50)

        # Save results
        output_dir = Path(config.paths.output_dir) / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)

        import json

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_dir / 'results.json'}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def interactive_inference(model: LoRAModel, config: DictConfig):
    """Interactive inference mode."""
    logger.info("Starting interactive inference mode...")
    logger.info("Type 'quit' to exit.")

    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                break

            if not prompt:
                continue

            # Tokenize input
            inputs = model.get_tokenizer()(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            inputs = {k: v.to(model.get_model().device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=config.inference.max_new_tokens,
                    temperature=config.inference.temperature,
                    top_p=config.inference.top_p,
                    top_k=config.inference.top_k,
                    do_sample=config.inference.do_sample,
                    pad_token_id=model.get_tokenizer().pad_token_id,
                    eos_token_id=model.get_tokenizer().eos_token_id,
                )

            # Decode response
            response = model.get_tokenizer().decode(
                outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
            )

            print(f"\nResponse: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during inference: {e}")

    logger.info("Interactive inference ended.")


if __name__ == "__main__":
    main()
