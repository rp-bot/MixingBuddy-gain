"""
Simple generation script for instrument classification.
Uses the trained model to generate predictions on test data.
"""

import sys
from pathlib import Path
import torch
import logging
import json
from transformers import AutoTokenizer, BitsAndBytesConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.simple_dataset import SimpleInstrumentDataset
from src.models.modular_multimodal_model import ModularMultimodalModel
from src.models.checkpoint_loading import load_trained_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_simple_model(checkpoint_path: str = "latest"):
    """Load a trained model for generation."""
    logger.info("Loading trained model...")

    model_name = "Qwen/Qwen2-7B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM with QLoRA quantization (same as training)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    from transformers import AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    # Freeze LLM parameters
    for param in llm.parameters():
        param.requires_grad = False

    # Configure MERT encoder (same as training)
    encoder_config = {
        "model_name": "m-a-p/MERT-v1-330M",
        "freeze": True,
        "device": llm.device,
        "input_sample_rate": 32000,
    }

    # Configure MLP projection (same as training)
    projection_config = {
        "type": "mlp",
        "hidden_dims": [2048, 4096, 4096, 2048],
        "activation": "relu",
        "dropout": 0.1,
        "use_layer_norm": True,
        "use_residual": False,
        "use_auxiliary_loss": True,
        "auxiliary_loss_weight": 0.05,
    }

    # Create model
    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=encoder_config,
        projection_config=projection_config,
    )

    # Load trained projection weights
    if checkpoint_path == "latest":
        # Find the latest checkpoint
        checkpoint_dir = PROJECT_ROOT / "outputs" / "simple_training" / "checkpoint-700"
        if not checkpoint_dir.exists():
            raise ValueError(f"No checkpoint directory found at {checkpoint_dir}")

        # Look for audio_projection.bin
        projection_path = checkpoint_dir / "audio_projection.bin"
        if not projection_path.exists():
            raise ValueError(f"No audio_projection.bin found at {projection_path}")

        logger.info(f"Loading projection weights from {projection_path}")
        state_dict = torch.load(projection_path, map_location="cpu")
        model.audio_projection.load_state_dict(state_dict)

        # Load MERT encoder weights if available
        mert_path = checkpoint_dir / "mert_encoder.bin"
        if mert_path.exists():
            logger.info(f"Loading MERT encoder weights from {mert_path}")
            mert_state_dict = torch.load(mert_path, map_location="cpu")
            model.audio_encoder.load_state_dict(mert_state_dict)
        else:
            logger.info("No MERT encoder weights found, using default weights")
    else:
        # Load from specific checkpoint
        projection_path = Path(checkpoint_path) / "audio_projection.bin"
        if not projection_path.exists():
            raise ValueError(f"No audio_projection.bin found at {projection_path}")

        logger.info(f"Loading projection weights from {projection_path}")
        state_dict = torch.load(projection_path, map_location="cpu")
        model.audio_projection.load_state_dict(state_dict)

        # Load MERT encoder weights if available
        mert_path = Path(checkpoint_path) / "mert_encoder.bin"
        if mert_path.exists():
            logger.info(f"Loading MERT encoder weights from {mert_path}")
            mert_state_dict = torch.load(mert_path, map_location="cpu")
            model.audio_encoder.load_state_dict(mert_state_dict)
        else:
            logger.info("No MERT encoder weights found, using default weights")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Model loaded successfully")
    return model


def generate_single_prediction(
    model, audio, instruction, max_new_tokens=50, use_noise=False, sample_rate=32000
):
    """Generate a single prediction for given audio and instruction."""
    with torch.no_grad():
        if use_noise:
            # Generate random noise instead of real audio
            audio = torch.randn(
                sample_rate * 3, generator=torch.Generator().manual_seed(42)
            )  # 3 seconds of random noise with fixed seed
            logger.info("Using random noise instead of real audio")

        # Prepare audio (ensure it's the right shape and dtype)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float()  # Ensure float32 for MERT

        # Generate response
        response = model.generate(
            text_input=instruction,
            audio=audio,
            max_new_tokens=max_new_tokens,
            system_message="",  # No system message for simple generation
        )

        return response


def main():
    """Main generation function."""

    # Configuration
    checkpoint_path = "latest"  # or path to specific checkpoint
    test_jsonl = PROJECT_ROOT / "data" / "inst_classify_test.jsonl"
    audio_root = PROJECT_ROOT / "data"
    sample_rate = 32000
    max_new_tokens = 50
    num_samples = 30  # Test with just 10 samples
    use_noise = False  # Set to True to use random noise instead of real audio

    # Load model
    model = load_simple_model(checkpoint_path)

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = SimpleInstrumentDataset(
        jsonl_path=test_jsonl,
        audio_root=audio_root,
        sample_rate=sample_rate,
        limit=num_samples,
    )

    logger.info(f"Generating predictions for {len(test_dataset)} samples...")

    # Generate predictions
    predictions = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        audio = sample["audio"]
        instruction = sample["instruction"]
        true_response = sample["response"]

        logger.info(f"Processing sample {i + 1}/{len(test_dataset)}")

        try:
            # Generate prediction
            predicted_response = generate_single_prediction(
                model, audio, instruction, max_new_tokens, use_noise, sample_rate
            )

            predictions.append(
                {
                    "sample_idx": i,
                    "instruction": instruction,
                    "true_response": true_response,
                    "predicted_response": predicted_response,
                }
            )

        except Exception as e:
            logger.error(f"Error generating prediction for sample {i}: {e}")
            predictions.append(
                {
                    "sample_idx": i,
                    "instruction": instruction,
                    "true_response": true_response,
                    "predicted_response": f"ERROR: {str(e)}",
                }
            )

    # Save results as JSON
    output_dir = PROJECT_ROOT / "outputs" / "simple_generation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / (
        "predictions_noise.json" if use_noise else "predictions.json"
    )
    with open(results_file, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
