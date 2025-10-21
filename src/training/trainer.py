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

        # Always use the global step for consistent logging
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
