"""
This module contains the LoRATrainer class for fine-tuning models with LoRA.
"""

import datetime
import math
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

    def __init__(self, experiment_tracker: Optional[ExperimentTracker]):
        self.experiment_tracker = experiment_tracker
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.experiment_tracker or not logs:
            return

        self.step = state.global_step
        self.experiment_tracker.log_metrics(logs, step=self.step)

    def on_save(self, args, state, control, **kwargs):
        if not self.experiment_tracker:
            return

        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            self.experiment_tracker.log_artifacts(checkpoint_dir)
        elif not torch.distributed.is_initialized():
            self.experiment_tracker.log_artifacts(checkpoint_dir)


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
            experiment_tracker: The experiment tracker to use.
        """
        self.model = model
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.trainer: Optional[Trainer] = None

    def setup_training_args(self) -> TrainingArguments:
        """Sets up the training arguments."""
        training_args_config = self.config.training.training_args
        if "run_name" not in training_args_config or not training_args_config.run_name:
            run_name = f"lora-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        else:
            run_name = training_args_config.run_name

        return TrainingArguments(
            output_dir=training_args_config.output_dir,
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
        if self.config.training.early_stopping.enabled:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping.patience
                )
            )
        if self.experiment_tracker:
            callbacks.append(ExperimentTrackingCallback(self.experiment_tracker))
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

        if self.experiment_tracker:
            self.experiment_tracker.log_params(self.config)

        self.trainer.train()

        if self.experiment_tracker:
            metrics = self.trainer.state.log_history[-1]
            self.experiment_tracker.log_metrics(metrics)

        return self.trainer

    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """
        Evaluates the model.

        Args:
            eval_dataset: The evaluation dataset.

        Returns:
            A dictionary of evaluation metrics.
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")
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
