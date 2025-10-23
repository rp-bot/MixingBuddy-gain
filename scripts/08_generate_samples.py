"""
Qualitative evaluation script for generating and inspecting model outputs.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import torch
import gc
import json
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    find_latest_checkpoint,
    initialize_tokenizer,
    load_dataset,
)


def load_trained_model(cfg: DictConfig) -> ModularMultimodalModel:
    """Load a trained model for evaluation."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    print("Loading trained model for generation...")

    # Load tokenizer
    tokenizer = initialize_tokenizer(cfg.model.model_name)

    # Determine checkpoint path
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint found when checkpoint_path is set to 'latest'"
            )

    # Load base model without LoRA first
    if cfg.model.use_qlora:
        print("Loading base model with QLoRA quantization...")
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
        llm = prepare_model_for_kbit_training(llm)
    else:
        print("Loading base model without quantization...")
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name,
            torch_dtype="auto",
        )

    # Load the trained LoRA weights if checkpoint path is provided
    if checkpoint_path:
        print(f"Loading LoRA weights from {checkpoint_path}")
        llm = PeftModel.from_pretrained(llm, checkpoint_path)

    # Initialize the multimodal model
    model = ModularMultimodalModel(
        model_name=cfg.model.model_name,
        use_qlora=cfg.model.use_qlora,
        lora_config=None,
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),
    )

    # The audio projection weights are a required component for evaluation.
    projection_path = f"{checkpoint_path}/audio_projection.bin"
    print(f"Loading audio projection weights from {projection_path}...")
    map_location = cfg.evaluation.model_loading.map_location
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(projection_path, map_location=map_location)

    # --- DEBUGGING PRINT ---
    print(f"Loaded audio_projection state_dict with keys: {state_dict.keys()}")
    # --- END DEBUGGING PRINT ---

    model.audio_projection.load_state_dict(
        state_dict,
        strict=cfg.evaluation.model_loading.strict_loading,
    )

    # Freeze all parameters for evaluation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Model loaded for generation.")
    model.print_trainable_parameters()

    gc.collect()
    torch.cuda.empty_cache()

    return model


def generate_and_compare(
    model: ModularMultimodalModel,
    dataset,
    num_samples: int,
    max_new_tokens: int,
    use_instruction: bool,
    system_message: str,
    output_dir: Path,
):
    """Generate responses for samples and save to JSONL for later analysis.

    Args:
        model: The multimodal model
        dataset: Dataset to generate from
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens to generate
        use_instruction: Whether to use instruction text
        system_message: System message for generation
        output_dir: Output directory for saving predictions
    """
    print("\n" + "=" * 50)
    print("GENERATING SAMPLES FOR QUALITATIVE EVALUATION")
    print("=" * 50 + "\n")

    predictions = []

    # Helper function to get expected magnitude range
    def get_expected_magnitude_range(error_category):
        """Get expected magnitude range based on error category."""
        if error_category == "no_error":
            return None, None
        elif error_category in ["quiet", "loud"]:
            return 3.0, 6.0
        elif error_category in ["very_quiet", "very_loud"]:
            return 6.0, 12.0
        return None, None

    # Handle None case for num_samples (generate for all samples in dataset)
    total_samples = num_samples if num_samples is not None else len(dataset)

    for i in tqdm(range(total_samples), desc="Generating predictions"):
        sample = dataset[i]
        instruction = sample["instruction"]
        audio = sample["audio"]
        ground_truth = sample["response"]
        global_uid = sample["global_uid"]
        target_stem = sample["target_stem"]
        error_category = sample["error_category"]

        # Get intended gain and timing info from the dataset item if available
        # Note: This field might not be directly in the sample, but in metadata
        # We'll extract it from the dataset's underlying data
        item_data = dataset.data[i]
        intended_gain_db = item_data["meta"].get("intended_gain_db", 0.0)
        start_sec = item_data["meta"].get("time_ref", {}).get("start_sec", 0.0)
        end_sec = item_data["meta"].get("time_ref", {}).get("end_sec", 0.0)

        # Get expected magnitude range
        expected_min_db, expected_max_db = get_expected_magnitude_range(error_category)

        # Determine the text input for the model based on the config
        text_for_generation = instruction if use_instruction else ""

        generated_text = model.generate(
            text_input=text_for_generation,
            audio=audio,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
        )

        # Store prediction with metadata
        prediction = {
            "global_uid": global_uid,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "generated": generated_text,
            "target_stem": target_stem,
            "error_category": error_category,
            "intended_gain_db": intended_gain_db,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "expected_magnitude_min_db": expected_min_db,
            "expected_magnitude_max_db": expected_max_db,
        }
        predictions.append(prediction)

        # Print first few samples for inspection
        if i < 3:
            print(f"\n--- Sample {i + 1}/{total_samples} ---")
            print(f"Global UID: {global_uid}")
            print(f"Target Stem: {target_stem} | Error: {error_category}")
            print(f"Timing: {start_sec:.1f}s - {end_sec:.1f}s")
            print(
                f"Instruction: {text_for_generation[:100]}..."
                if len(text_for_generation) > 100
                else f"Instruction: {text_for_generation}"
            )
            print(f"\nGround Truth: {ground_truth}")
            print(f"\nGenerated: {generated_text}")
            print("-" * 50)

    # Save predictions to JSONL
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / "predictions.jsonl"

    print(f"\nSaving predictions to {predictions_file}")
    with open(predictions_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Saved {len(predictions)} predictions.")
    return predictions_file


@hydra.main(config_path="../configs", config_name="03_evalutate_synthesis_instructions", version_base=None)
def main(cfg: DictConfig):
    """
    Main generation function.
    """
    # Load model
    model = load_trained_model(cfg)

    # Load test dataset
    limit = cfg.evaluation.num_generation_samples
    test_dataset = load_dataset(cfg, "test", limit=limit)

    # Get generation parameters from config
    max_new_tokens = cfg.evaluation.max_new_tokens
    use_instruction = cfg.data.use_instructions
    system_message = cfg.data.system_message

    # Get output directory from config - use the same run name as the checkpoint
    # Extract run name from checkpoint path
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path == "latest":
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint found when checkpoint_path is set to 'latest'"
            )

    # Extract run name from checkpoint path
    # Expected format: outputs/checkpoints/mixing_buddy_milestone_0/{run_name}/checkpoint-{step}
    checkpoint_path = Path(checkpoint_path)
    # Handle both cases: checkpoint at root or in subdirectory
    if checkpoint_path.name.startswith("checkpoint-"):
        # Checkpoint is at root level, use the parent directory name
        run_name = checkpoint_path.parent.name
    else:
        # Checkpoint is in a subdirectory, use the checkpoint directory name
        run_name = checkpoint_path.name

    # Create evaluation directory structure
    output_dir = Path("outputs/evaluation") / run_name

    # Generate and compare (limit can be None to generate for all samples)
    predictions_file = generate_and_compare(
        model,
        test_dataset,
        num_samples=limit,  # Can be None to generate for all samples
        max_new_tokens=max_new_tokens,
        use_instruction=use_instruction,
        system_message=system_message,
        output_dir=output_dir,
    )

    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print(f"Predictions saved to: {predictions_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
