"""
Qualitative evaluation script for generating and inspecting model outputs.
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
):
    """Generate responses for a few samples and compare with ground truth."""
    print("\n" + "=" * 50)
    print("GENERATING SAMPLES FOR QUALITATIVE EVALUATION")
    print("=" * 50 + "\n")

    for i in range(num_samples):
        sample = dataset[i]
        instruction = sample["instruction"]
        audio = sample["audio"]
        ground_truth = sample["response"]

        # Determine the text input for the model based on the config
        text_for_generation = instruction if use_instruction else ""

        print(f"--- Sample {i + 1}/{num_samples} ---")
        print(f"Instruction:\n{text_for_generation}")
        print(f"\nGround Truth Response:\n{ground_truth}")

        generated_text = model.generate(
            text_input=text_for_generation,
            audio=audio,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
        )

        print(f"\nModel Generated Response:\n{generated_text}")
        print("-" * 50 + "\n")


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
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

    # Generate and compare
    generate_and_compare(
        model,
        test_dataset,
        num_samples=limit,
        max_new_tokens=max_new_tokens,
        use_instruction=use_instruction,
        system_message=system_message,
    )


if __name__ == "__main__":
    main()
