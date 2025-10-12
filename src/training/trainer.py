"""
This module contains the LoRATrainer class for fine-tuning models with LoRA.
"""

import math
import os
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.collator import MultimodalDataCollator
from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.experiment_tracking import ExperimentTracker


class ExperimentTrackingCallback(TrainerCallback):
    """Callback for logging metrics and artifacts to an experiment tracker."""

    def __init__(
        self,
        experiment_tracker: Optional[ExperimentTracker],
        model: Optional[ModularMultimodalModel] = None,
    ):
        self.experiment_tracker = experiment_tracker
        self.model = model
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker is required for logging")
        if not logs:
            return

        self.step = state.global_step
        self.experiment_tracker.log_metrics(logs, step=self.step)

    def on_save(self, args, state, control, **kwargs):
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker is required for artifact logging")

        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"

        # Save custom model components in checkpoint directory
        # if self.model is not None:
        self._save_custom_components(checkpoint_dir)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            self.experiment_tracker.log_artifacts(checkpoint_dir)
        elif not torch.distributed.is_initialized():
            self.experiment_tracker.log_artifacts(checkpoint_dir)

    def _save_custom_components(self, checkpoint_dir: str):
        """Save custom model components that aren't handled by HuggingFace Trainer."""
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save audio projection weights
        torch.save(
            self.model.audio_projection.state_dict(),
            f"{checkpoint_dir}/audio_projection.bin",
        )

        # Save LoRA adapter files (these are needed for inference)
        # Save the LoRA adapter to the checkpoint directory
        self.model.llm.save_pretrained(checkpoint_dir)

        print(f"Saved custom components to {checkpoint_dir}/")


def compute_metrics(eval_pred):
    """Computes perplexity and loss for evaluation."""
    logits, labels = eval_pred
    # TODO: this might need adjustment based on the actual output format.
    # It assumes logits are returned for all tokens.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    perplexity = math.exp(loss.item())
    return {"perplexity": perplexity, "loss": loss.item()}


class LoRATrainer:
    """
    A trainer for fine-tuning models with LoRA.
    """

    def __init__(
        self,
        model: ModularMultimodalModel,
        config: DictConfig,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        """
        Initializes the LoRATrainer.

        Args:
            model: The model to train.
            config: The configuration for training.
            experiment_tracker: The experiment tracker to use (optional for evaluation).

        Raises:
            ValueError: If experiment_tracker is None during training.
        """
        # Only require tracker for training, not evaluation
        self.is_training = False

        self.model = model
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.trainer: Optional[Trainer] = None

    def setup_training_args(self) -> TrainingArguments:
        """Sets up the training arguments."""
        training_args_config = self.config.training.training_args

        # Get run name from experiment tracker (should always be set during initialization)
        run_name = self.experiment_tracker._current_run_name

        # Create output directory if it doesn't exist
        output_dir = os.path.join(training_args_config.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_args_config.num_train_epochs,
            per_device_train_batch_size=training_args_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_args_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args_config.gradient_accumulation_steps,
            learning_rate=training_args_config.learning_rate,
            weight_decay=training_args_config.weight_decay,
            warmup_ratio=training_args_config.warmup_ratio,
            lr_scheduler_type=training_args_config.lr_scheduler_type,
            logging_steps=training_args_config.logging_steps,
            save_steps=training_args_config.save_steps,
            save_total_limit=training_args_config.save_total_limit,
            eval_strategy=training_args_config.eval_strategy,
            eval_steps=training_args_config.eval_steps,
            load_best_model_at_end=training_args_config.load_best_model_at_end,
            metric_for_best_model=training_args_config.metric_for_best_model,
            greater_is_better=training_args_config.greater_is_better,
            report_to=training_args_config.report_to,
            run_name=run_name,
            fp16=self.config.training.mixed_precision.enabled
            and self.config.training.mixed_precision.dtype == "fp16",
            bf16=self.config.training.mixed_precision.enabled
            and self.config.training.mixed_precision.dtype == "bf16",
            seed=self.config.env.seed,
            dataloader_pin_memory=False,  # Reduce memory usage
            dataloader_num_workers=0,  # Reduce memory usage
            remove_unused_columns=False,  # Keep all columns
        )

    def setup_data_collator(self) -> MultimodalDataCollator:
        """Sets up the data collator."""
        return MultimodalDataCollator(
            tokenizer=self.model.tokenizer,
            pad_to_multiple_of=8,  # For efficiency with tensor cores
        )

    def setup_callbacks(self) -> List[TrainerCallback]:
        """Sets up the callbacks."""
        callbacks = []

        # Early stopping callback (optional)
        if self.config.training.early_stopping.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping.patience
                )
            )

        # Experiment tracking callback (only for training)
        if self.experiment_tracker:
            callbacks.append(
                ExperimentTrackingCallback(self.experiment_tracker, self.model)
            )

        return callbacks

    def train(self, train_dataset, eval_dataset: Optional[Any] = None) -> Trainer:
        """
        Trains the model.

        Args:
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.

        Returns:
            The trainer object.
        """
        if not self.experiment_tracker:
            raise ValueError("Experiment tracker is required for training")

        self.is_training = True
        training_args = self.setup_training_args()
        data_collator = self.setup_data_collator()
        callbacks = self.setup_callbacks()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # Log configuration parameters (required)
        self.experiment_tracker.log_params(self.config)

        self.trainer.train()

        # Log final metrics (required)
        metrics = self.trainer.state.log_history[-1]
        self.experiment_tracker.log_metrics(metrics)

        return self.trainer

    def _setup_evaluation_trainer(self):
        """Sets up the trainer for evaluation only."""
        # Create minimal training args for evaluation
        from transformers import TrainingArguments

        # Create a temporary output directory for evaluation
        import tempfile

        temp_dir = tempfile.mkdtemp()

        # Get batch size from evaluation config or use training config
        eval_batch_size = self.config.evaluation.batch_size
        

        training_args = TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=eval_batch_size,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            fp16=self.config.training.mixed_precision.enabled
            and self.config.training.mixed_precision.dtype == "fp16",
            bf16=self.config.training.mixed_precision.enabled
            and self.config.training.mixed_precision.dtype == "bf16",
        )

        data_collator = self.setup_data_collator()
        callbacks = self.setup_callbacks()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=None,  # Will be set during evaluation
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """
        Evaluates the model.

        Args:
            eval_dataset: The evaluation dataset.

        Returns:
            A dictionary of evaluation metrics.
        """
        if not self.trainer:
            # Initialize trainer for evaluation if not already done
            self._setup_evaluation_trainer()
        return self.trainer.evaluate(eval_dataset)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saves the model.

        Args:
            output_dir: The directory to save the model to.
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")

        output_dir = output_dir or self.trainer.args.output_dir
        self.model.llm.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)
        # Save projection layer
        torch.save(
            self.model.audio_projection.state_dict(),
            f"{output_dir}/audio_projection.bin",
        )

        # Log artifacts (only if tracker is available)
        if self.experiment_tracker:
            self.experiment_tracker.log_artifacts(output_dir)

    def get_trainable_parameters_info(self) -> Dict[str, Any]:
        """
        Gets information about the trainable parameters.

        Returns:
            A dictionary with trainable parameter information.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "total": all_param,
            "trainable": trainable_params,
            "percentage": 100 * trainable_params / all_param,
        }
