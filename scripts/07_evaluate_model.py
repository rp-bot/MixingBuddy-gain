"""
Model evaluation script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path
import gc
import torch
from datasets import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.data.collator import MultimodalDataCollator  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    find_latest_checkpoint,
    initialize_tokenizer,
    load_dataset,
)


def load_trained_model(cfg: DictConfig):
    """Load a trained model for evaluation."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    print("Loading trained model for evaluation...")

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
        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg.model.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.model.quantization.bnb_4bit_compute_dtype
            ),
            bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
        )

        # Load model with quantization
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )

        # Prepare for k-bit training
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
        lora_config=None,  # LoRA is already applied to the model
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),  # Pass encoder configuration
    )

    # The audio projection weights are a required component for evaluation.
    projection_path = f"{checkpoint_path}/audio_projection.bin"
    print(f"Loading audio projection weights from {projection_path}...")

    map_location = cfg.evaluation.model_loading.map_location
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(projection_path, map_location=map_location)
    model.audio_projection.load_state_dict(
        state_dict,
        strict=cfg.evaluation.model_loading.strict_loading,
    )

    # Freeze all parameters for evaluation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

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


def load_test_dataset(cfg: DictConfig):
    """Load test dataset for evaluation."""
    # Use evaluation-specific limit if set, otherwise use data limit
    limit = (
        cfg.evaluation.max_samples
        if cfg.evaluation.max_samples
        else cfg.data.get("limit")
    )

    # Load the PyTorch dataset
    pytorch_dataset = load_dataset(cfg, dataset_type="test", limit=limit)

    # Convert to HuggingFace Dataset for SFTTrainer
    test_dataset = Dataset.from_list(
        [pytorch_dataset[i] for i in range(len(pytorch_dataset))]
    )

    return test_dataset


def run_evaluation(cfg: DictConfig):
    """Run evaluation on the test dataset."""
    print("Starting evaluation...")

    # Initialize model
    model = load_trained_model(cfg)

    # Load test dataset
    test_dataset = load_test_dataset(cfg)

    print("Setting up evaluation with SFTTrainer...")

    # A temporary output dir is required by the Trainer, but it won't be used for much.
    output_dir = Path(cfg.evaluation.get("output_dir", "temp_eval_outputs"))
    output_dir.mkdir(exist_ok=True, parents=True)

    eval_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=cfg.evaluation.batch_size,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,  # Keep all columns for our model
        label_names=[
            "labels"
        ],  # Specify that 'labels' is a label field for loss computation
    )

    # The stride of the audio encoder is needed to correctly pad the text tokens
    audio_encoder_stride = model.audio_encoder.model.config.hop_length

    # Create data collator
    data_collator = MultimodalDataCollator(
        tokenizer=model.tokenizer,
        pad_to_multiple_of=8,
        audio_encoder_stride=audio_encoder_stride,
    )

    # Initialize the SFTTrainer for evaluation (provides additional metrics like entropy and token accuracy)
    # Note: SFTTrainer requires a train_dataset even for eval-only usage, so we pass test_dataset as both
    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=test_dataset,  # Required by SFTTrainer even for eval-only
        eval_dataset=test_dataset,
        processing_class=model.tokenizer,  # Pass tokenizer to SFTTrainer
        data_collator=data_collator,
    )

    # Run evaluation
    print("Running evaluation on test dataset...")
    test_results = trainer.evaluate()
    print(f"Evaluation results: {test_results}")

    # Save predictions if configured
    if cfg.evaluation.save_predictions:
        save_predictions(cfg, test_results)

    return test_results


def save_predictions(cfg: DictConfig, results: dict):
    """Save evaluation results to file."""
    import json

    print("Saving evaluation results...")

    predictions_dir = Path(cfg.evaluation.predictions_output_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save the evaluation results
    results_file = predictions_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {results_file}")


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
