"""
Main training script for the multimodal model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
import torch
import logging
from transformers import TrainingArguments, EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer

# Configure logging to work better with tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collator import MultimodalDataCollator  # noqa: E402
from src.data.dataset import MixingDataset  # noqa: E402
from src.models.modular_multimodal_model import ModularMultimodalModel  # noqa: E402
from src.training.trainer import ExperimentTrackingCallback  # noqa: E402
from src.utils.model_utils import (  # noqa: E402
    create_lora_config,
    initialize_lora_model,
    initialize_qlora_model,
    initialize_tokenizer,
    initialize_experiment_tracker,
    IterableDatasetWrapper,
)


class ProjectionDiagnosticCallback(TrainerCallback):
    """Monitor projection layer during training to diagnose issues."""

    def __init__(self, model):
        self.model = model
        self.step = 0
        self.grad_history = []
        self.weight_history = []
        self.optimizer_checked = False

    def on_train_begin(self, args, state, control, model, **kwargs):
        """Check optimizer when training begins."""
        if not self.optimizer_checked:
            # Try to get optimizer from kwargs
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                # Optimizer may not be in kwargs, try to check on first step instead
                print("\n" + "=" * 60)
                print("OPTIMIZER DIAGNOSTIC")
                print("=" * 60)
                print("Note: Optimizer not yet available in callback.")
                print("Will check on first training step.")
                print("=" * 60 + "\n")
                return

            print("\n" + "=" * 60)
            print("OPTIMIZER DIAGNOSTIC")
            print("=" * 60)
            proj_param_ids = {id(p) for p in self.model.audio_projection.parameters()}
            optimizer_param_ids = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }
            proj_in_optimizer = proj_param_ids.issubset(optimizer_param_ids)
            print(f"Projection parameters in optimizer: {proj_in_optimizer}")
            print(f"Projection param count: {len(proj_param_ids)}")
            print(f"Optimizer param count: {len(optimizer_param_ids)}")

            # If not in optimizer, let's check why
            if not proj_in_optimizer:
                print("\n⚠️  WARNING: Projection parameters NOT in optimizer!")
                print("Checking requires_grad status:")
                for name, param in self.model.audio_projection.named_parameters():
                    print(f"  {name}: requires_grad={param.requires_grad}")

            print("=" * 60 + "\n")
            self.optimizer_checked = True

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1

        # Check optimizer on first step if we haven't checked it yet
        if self.step == 1 and not self.optimizer_checked:
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                print("\n" + "=" * 60)
                print("OPTIMIZER DIAGNOSTIC (First Step)")
                print("=" * 60)
                proj_param_ids = {
                    id(p) for p in self.model.audio_projection.parameters()
                }
                optimizer_param_ids = {
                    id(p) for group in optimizer.param_groups for p in group["params"]
                }
                proj_in_optimizer = proj_param_ids.issubset(optimizer_param_ids)
                print(f"Projection parameters in optimizer: {proj_in_optimizer}")
                print(f"Projection param count: {len(proj_param_ids)}")
                print(f"Optimizer param count: {len(optimizer_param_ids)}")

                if not proj_in_optimizer:
                    print("\n⚠️  WARNING: Projection parameters NOT in optimizer!")
                    print("Checking requires_grad status:")
                    for name, param in self.model.audio_projection.named_parameters():
                        print(f"  {name}: requires_grad={param.requires_grad}")

                print("=" * 60 + "\n")
                self.optimizer_checked = True

        # NOTE: on_step_end is called AFTER optimizer.step() and zero_grad()
        # So gradients will always be None here. Use on_optimizer_step instead.

        # Check weight changes (compare to first step)
        if self.step == 1:
            self.initial_weights = {
                name: param.data.clone()
                for name, param in self.model.audio_projection.named_parameters()
            }
        elif self.step % 5 == 0:
            weight_changes = []
            for name, param in self.model.audio_projection.named_parameters():
                if name in self.initial_weights:
                    change = (
                        (param.data - self.initial_weights[name]).abs().mean().item()
                    )
                    weight_changes.append(change)
            avg_change = (
                sum(weight_changes) / len(weight_changes) if weight_changes else 0.0
            )
            self.weight_history.append((self.step, avg_change))

            print(f"\n[DIAGNOSTIC - WEIGHT CHANGES] Step {self.step}:")
            print(f"  Avg weight change from step 1: {avg_change:.8f}")

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Check gradients BEFORE optimizer applies and zeros them."""
        current_step = state.global_step + 1  # Will be incremented after this

        # Check gradients BEFORE they're applied and zeroed
        has_grad = False
        grad_values = []
        for name, param in self.model.audio_projection.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_values.append(param.grad.abs().mean().item())

        avg_grad = sum(grad_values) / len(grad_values) if grad_values else 0.0
        self.grad_history.append(avg_grad)

        # Print on first step and every 5 steps
        if current_step == 1 or current_step % 5 == 0:
            print(f"\n[DIAGNOSTIC - GRADIENTS] Step {current_step}:")
            print(f"  Gradients present: {has_grad}")
            print(f"  Avg gradient magnitude: {avg_grad:.8f}")

            # Show gradient improvement from baseline
            if len(self.grad_history) > 1:
                initial_grad = self.grad_history[0]
                if initial_grad > 0:
                    improvement = (avg_grad / initial_grad) * 100
                    print(f"  Gradient magnitude vs step 1: {improvement:.1f}%")

            if has_grad and avg_grad < 1e-6:
                print("  ⚠️  WARNING: Gradients are extremely small!")


def initialize_model_and_tokenizer(cfg: DictConfig):
    """Initialize model, tokenizer, and LoRA configuration."""
    print("Initializing model and tokenizer...")
    tokenizer = initialize_tokenizer(cfg.model.model_name)
    lora_config = create_lora_config(cfg)

    if cfg.model.use_qlora:
        llm = initialize_qlora_model(cfg, lora_config, tokenizer)
    else:
        llm = initialize_lora_model(cfg, lora_config, tokenizer)

    model = ModularMultimodalModel(
        llm=llm,
        tokenizer=tokenizer,
        encoder_config=cfg.model.get("encoder"),
        projection_config=cfg.model.get("projection"),
    )
    print("Model and tokenizer initialized.")
    model.print_trainable_parameters()
    return model, tokenizer


def load_datasets(cfg: DictConfig, tokenizer):
    """Load and split train, validation, and test datasets."""
    print("Loading data...")

    # Expected audio length: 10 seconds at 32kHz = 320,000 samples
    audio_length = int(cfg.data.chunk.sec * cfg.data.audio.sample_rate)
    print(
        f"Expected audio length: {audio_length} samples ({cfg.data.chunk.sec}s at {cfg.data.audio.sample_rate}Hz)"
    )

    full_train_pytorch_dataset = MixingDataset(
        jsonl_path=cfg.data.train_jsonl_path,
        audio_root=cfg.data.train_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )
    # Wrap in our custom wrapper to avoid HuggingFace Dataset truncation
    # This keeps data in PyTorch format and loads audio on-demand
    full_train_dataset = IterableDatasetWrapper(full_train_pytorch_dataset)

    test_pytorch_dataset = MixingDataset(
        jsonl_path=cfg.data.test_jsonl_path,
        audio_root=cfg.data.test_audio_root,
        sample_rate=cfg.data.audio.sample_rate,
        limit=cfg.data.limit,
        use_instructions=cfg.data.use_instructions,
        system_message=cfg.data.system_message,
    )
    # Wrap in our custom wrapper
    test_dataset = IterableDatasetWrapper(test_pytorch_dataset)

    train_val_split = full_train_dataset.train_test_split(
        test_size=0.2, seed=cfg.env.seed
    )
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    print(
        f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


@hydra.main(
    config_path="../configs",
    config_name="07_train_all_modules",
    version_base=None,
)
def main(cfg: DictConfig):
    """Main training function."""
    tracker = initialize_experiment_tracker(cfg, required=False)
    model, tokenizer = initialize_model_and_tokenizer(cfg)

    # === DIAGNOSTIC: Check projection layer parameter status ===
    print("\n" + "=" * 60)
    print("PROJECTION LAYER DIAGNOSTIC")
    print("=" * 60)
    proj_params = list(model.audio_projection.parameters())
    proj_trainable = [p for p in proj_params if p.requires_grad]
    print(f"Total projection parameters: {len(proj_params)}")
    print(f"Trainable projection parameters: {len(proj_trainable)}")
    print(
        f"All projection params require_grad: {all(p.requires_grad for p in proj_params)}"
    )

    # Check if projection is in model's trainable parameters
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    proj_in_trainable = any(
        id(p) in [id(tp) for tp in proj_trainable] for p in all_trainable
    )
    print(f"Projection params in model.parameters(): {proj_in_trainable}")
    print("=" * 60 + "\n")

    train_dataset, val_dataset, test_dataset = load_datasets(cfg, tokenizer)

    # Convert Hydra config to a dictionary to safely modify it
    training_args_dict = OmegaConf.to_container(
        cfg.training.training_args, resolve=True
    )
    # Prevent SFTTrainer from removing custom columns like 'audio' and 'messages'
    training_args_dict["remove_unused_columns"] = False
    # Specify that 'labels' is a label field so the Trainer properly computes loss during evaluation
    training_args_dict["label_names"] = ["labels"]
    # Ensure tqdm progress bars work properly
    training_args_dict["disable_tqdm"] = False

    # Update output_dir to include run name for better organization
    if tracker:
        run_name = tracker._current_run_name
    else:
        from src.utils.model_utils import generate_run_name

        run_name = generate_run_name(cfg)

    base_output_dir = training_args_dict["output_dir"]
    training_args_dict["output_dir"] = f"{base_output_dir}/{run_name}"
    print(f"Checkpoints will be saved to: {training_args_dict['output_dir']}")

    training_args = TrainingArguments(**training_args_dict)

    callbacks = []
    # Always add ExperimentTrackingCallback to save LoRA adapters and audio projection
    # The callback handles None tracker gracefully
    callbacks.append(ExperimentTrackingCallback(tracker, model))
    # Add diagnostic callback to monitor projection layer training
    callbacks.append(ProjectionDiagnosticCallback(model))
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping.patience
            )
        )

    # The stride of the audio encoder is needed to correctly pad the text tokens
    # so that the sequence length is consistent for the trainer.
    audio_encoder_stride = model.audio_encoder.model.config.hop_length
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
        processing_class=tokenizer,  # Pass tokenizer to SFTTrainer
        # dataset_text_field is removed as the collator now handles all formatting.
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    print("(Optimizer diagnostic will be shown at step 0)\n")
    trainer.train()
    print("Training finished.")

    print("Saving final model...")
    final_model_dir = training_args.output_dir
    trainer.save_model(final_model_dir)

    # Save custom components (audio projection weights) with the final model
    print("Saving custom model components (audio projection)...")
    torch.save(
        model.audio_projection.state_dict(),
        f"{final_model_dir}/audio_projection.bin",
    )

    # Save PEFT adapter files to the final model directory
    print("Saving PEFT adapter files to final model directory...")
    model.llm.save_pretrained(final_model_dir)

    print(f"Model and custom components saved to {final_model_dir}")

    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()
