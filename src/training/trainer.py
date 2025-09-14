"""
Training utilities and trainer class for LoRA fine-tuning.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from src.models.lora_model import LoRAModel
from src.utils.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Custom trainer for LoRA fine-tuning with experiment tracking."""

    def __init__(
        self,
        model: LoRAModel,
        config: DictConfig,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        """
        Initialize LoRA trainer.

        Args:
            model: LoRA model instance
            config: Training configuration
            experiment_tracker: Optional experiment tracker
        """
        self.model = model
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.trainer = None

        # Set up paths
        self.output_dir = Path(config.paths.output_dir) / "checkpoints"
        self.logs_dir = Path(config.paths.logs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def setup_training_args(self) -> TrainingArguments:
        """Set up training arguments."""
        training_config = self.config.training.training_args

        # Generate run name if not provided
        run_name = training_config.get("run_name")
        if not run_name:
            model_name = self.config.model.pretrained_model_name_or_path.split("/")[-1]
            run_name = f"lora-{model_name}-{self.config.env.seed}"

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.lr_scheduler_type,
            logging_steps=training_config.logging_steps,
            eval_steps=training_config.eval_steps,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            eval_strategy=training_config.evaluation_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            report_to=training_config.report_to,
            run_name=run_name,
            seed=self.config.env.seed,
            data_seed=self.config.env.seed,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=self.config.data.dataloader.num_workers,
            fp16=self.config.env.mixed_precision == "fp16",
            bf16=self.config.env.mixed_precision == "bf16",
            gradient_checkpointing=self.config.model.training.gradient_checkpointing,
            optim=self.config.training.optimizer.type,
            adam_beta1=self.config.training.optimizer.adam_beta1,
            adam_beta2=self.config.training.optimizer.adam_beta2,
            adam_epsilon=self.config.training.optimizer.adam_epsilon,
            max_grad_norm=self.config.training.gradient_clipping.max_grad_norm,
            logging_dir=str(self.logs_dir),
            logging_first_step=True,
            logging_nan_inf_filter=True,
            save_safetensors=True,
            push_to_hub=False,
        )

        return args

    def setup_data_collator(self):
        """Set up data collator for language modeling."""
        tokenizer = self.model.get_tokenizer()
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficient GPU usage
        )

    def setup_callbacks(self) -> List:
        """Set up training callbacks."""
        callbacks = []

        # Early stopping
        if self.config.training.early_stopping.enabled:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping.patience,
                early_stopping_threshold=self.config.training.early_stopping.threshold,
            )
            callbacks.append(early_stopping)

        # Custom callback for experiment tracking
        if self.experiment_tracker:
            tracking_callback = ExperimentTrackingCallback(self.experiment_tracker)
            callbacks.append(tracking_callback)

        return callbacks

    def train(
        self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None
    ) -> Trainer:
        """Train the model."""
        logger.info("Setting up training...")

        # Get the PEFT model
        peft_model = self.model.get_model()

        # Set up training arguments
        training_args = self.setup_training_args()

        # Set up data collator
        data_collator = self.setup_data_collator()

        # Set up callbacks
        callbacks = self.setup_callbacks()

        # Create trainer
        self.trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Watch model for gradients (WandB only)
        if self.experiment_tracker and self.experiment_tracker.backend == "wandb":
            self.experiment_tracker.watch_model(peft_model)

        # Log training configuration
        if self.experiment_tracker:
            self.experiment_tracker.log_params(
                {
                    "model_name": self.config.model.pretrained_model_name_or_path,
                    "lora_r": self.config.model.lora.r,
                    "lora_alpha": self.config.model.lora.lora_alpha,
                    "lora_dropout": self.config.model.lora.lora_dropout,
                    "learning_rate": training_args.learning_rate,
                    "batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "num_epochs": training_args.num_train_epochs,
                    "max_length": self.config.data.processing.max_length,
                }
            )

        # Start training
        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Log final metrics
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(
                {
                    "train_loss": train_result.training_loss,
                    "train_runtime": train_result.metrics["train_runtime"],
                    "train_samples_per_second": train_result.metrics[
                        "train_samples_per_second"
                    ],
                    "train_steps_per_second": train_result.metrics[
                        "train_steps_per_second"
                    ],
                }
            )

        logger.info("Training completed!")
        return self.trainer

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Train the model first.")

        logger.info("Evaluating model...")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataloader.dataset)

        # Log evaluation metrics
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics(eval_results)

        logger.info(f"Evaluation results: {eval_results}")
        return eval_results

    def save_model(self, output_dir: Optional[Union[str, Path]] = None):
        """Save the trained model."""
        if output_dir is None:
            output_dir = self.output_dir / "final_model"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        self.model.save_model(output_dir)

        # Save training arguments
        if self.trainer is not None:
            self.trainer.save_state()

        # Log model artifacts
        if self.experiment_tracker:
            self.experiment_tracker.log_artifacts(str(output_dir), "final_model")

        logger.info(f"Model saved to {output_dir}")

    def get_trainable_parameters_info(self) -> Dict[str, Any]:
        """Get information about trainable parameters."""
        return self.model.get_trainable_parameters()


class ExperimentTrackingCallback:
    """Custom callback for experiment tracking."""

    def __init__(self, experiment_tracker: ExperimentTracker):
        self.experiment_tracker = experiment_tracker
        self.step = 0

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logs are written."""
        if logs and self.experiment_tracker:
            # Log metrics to experiment tracker
            self.experiment_tracker.log_metrics(logs, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        """Called when model is saved."""
        if self.experiment_tracker:
            # Log checkpoint artifacts
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_dir.exists():
                self.experiment_tracker.log_artifacts(
                    str(checkpoint_dir), f"checkpoint-{state.global_step}"
                )


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred

    # For causal language modeling, we typically compute perplexity
    # This is a simplified version - you might want to add more metrics

    # Shift predictions and labels for next token prediction
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross entropy loss
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Compute perplexity
    perplexity = torch.exp(loss)

    return {"perplexity": perplexity.item(), "loss": loss.item()}


def setup_optimizer_and_scheduler(
    model: PeftModel, config: DictConfig, num_training_steps: int
):
    """Set up optimizer and learning rate scheduler."""
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Set up optimizer
    optimizer = AdamW(
        trainable_params,
        lr=config.training.training_args.learning_rate,
        betas=(
            config.training.optimizer.adam_beta1,
            config.training.optimizer.adam_beta2,
        ),
        eps=config.training.optimizer.adam_epsilon,
        weight_decay=config.training.training_args.weight_decay,
    )

    # Set up scheduler
    num_warmup_steps = int(
        config.training.training_args.warmup_ratio * num_training_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler
