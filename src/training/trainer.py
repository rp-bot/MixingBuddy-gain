"""
This module contains the LoRATrainer class for fine-tuning models with LoRA.
"""

import math
import os
from typing import Any, Dict, List, Optional
import logging

import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_pt_utils import find_batch_size
from transformers.trainer_utils import EvalLoopOutput

from src.data.collator import MultimodalDataCollator
from src.models.modular_multimodal_model import ModularMultimodalModel
from src.utils.experiment_tracking import ExperimentTracker

logger = logging.getLogger(__name__)


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

        is_eval = any(key.startswith("eval_") for key in logs.keys())
        if is_eval:
            # For evaluation, don't pass a step to use wandb's internal step counter
            # This creates a separate x-axis for evaluation metrics
            self.experiment_tracker.log_metrics(logs)
        else:
            # For training, log with the global step
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


class MemorySafeTrainer(Trainer):
    """
    A Trainer subclass that moves predictions and labels to the CPU during evaluation
    to avoid out-of-memory errors with large datasets.
    """

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        An evaluation loop that moves predictions and labels to the CPU to save memory.
        """
        # This is a modified version of the original evaluation_loop in transformers.Trainer
        # The main change is moving tensors to the CPU after each step.
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, torch.utils.data.IterableDataset):
            logger.info(f"  Num examples = {num_examples}")
        else:
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do not use containers that move all tensors to the GPU at once
        all_losses = None

        for step, inputs in enumerate(tqdm(dataloader, desc=description)):
            loss, logits, labels = self.prediction_step(
                model,
                inputs,
                prediction_loss_only=False,  # Get logits for debugging
                ignore_keys=ignore_keys,
            )

            # --- DEBUGGING PRINTS ---
            if step == 0:
                print("\n--- INSIDE EVALUATION_LOOP (FIRST BATCH) ---")
                # Decode the first sample in the batch to see what's being evaluated
                input_ids = inputs["input_ids"][0]
                labels_vis = inputs["labels"][
                    0
                ].clone()  # clone to avoid modifying original

                # Decode input_ids, skipping pad tokens for readability
                decoded_input = self.tokenizer.decode(
                    input_ids, skip_special_tokens=True
                )
                print(f"Decoded Input: '{decoded_input}'")

                # Decode labels, replacing -100 with pad token before decoding
                labels_vis[labels_vis == -100] = self.tokenizer.pad_token_id
                decoded_labels = self.tokenizer.decode(
                    labels_vis, skip_special_tokens=True
                )
                print(f"Decoded Labels (target for loss): '{decoded_labels}'")

                print(f"Number of Tokens in Sequence: {len(input_ids)}")

                # Decode the model's prediction from the logits
                if logits is not None:
                    preds = torch.argmax(logits[0], dim=-1)
                    # Don't skip special tokens, to see exactly what's predicted
                    decoded_preds = self.tokenizer.decode(
                        preds, skip_special_tokens=True
                    )
                    print(f"Decoded Prediction from Logits: '{decoded_preds}'")

                print(f"Number of Tokens in Logits: {len(preds)}")
                print(f"Audio Tensor Shape: {inputs['audio'].shape}")
                print(f"Loss for first batch: {loss.item():.4f}")
                print("------------------------------------------\n")
            # --- END DEBUGGING PRINTS ---

            batch_size = find_batch_size(inputs)

            if loss is not None:
                losses = loss.repeat(batch_size)
                all_losses = (
                    losses
                    if all_losses is None
                    else torch.cat((all_losses, losses), dim=0)
                )

            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        metrics = {}
        if all_losses is not None:
            mean_loss = all_losses.mean().item()
            metrics[f"{metric_key_prefix}_loss"] = mean_loss
            try:
                perplexity = math.exp(mean_loss)
                metrics[f"{metric_key_prefix}_perplexity"] = perplexity
            except OverflowError:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None, label_ids=None, metrics=metrics, num_samples=num_examples
        )


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

        self.trainer = MemorySafeTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            callbacks=callbacks,
        )

        # Log configuration parameters (required)
        self.experiment_tracker.log_params(self.config)

        self.trainer.train()

        # Log final metrics (required) - This might be redundant with the callback
        # metrics = self.trainer.state.log_history[-1]
        # self.experiment_tracker.log_metrics(metrics)

        return self.trainer

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
