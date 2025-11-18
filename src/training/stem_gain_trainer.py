"""
Custom trainer for stem classification and gain regression.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import Trainer

from src.models.stem_gain_model import StemGainModel

logger = logging.getLogger(__name__)


class StemGainTrainer(Trainer):
    """Trainer for multi-task learning: stem classification + gain regression.
    
    Implements combined loss with configurable weights for classification and regression.
    """
    
    def __init__(
        self,
        model: StemGainModel,
        classification_weight: float = 1.0,
        regression_weight: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            model: StemGainModel instance
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
            **kwargs: Additional arguments passed to Trainer
        """
        super().__init__(model=model, **kwargs)
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        # Store outputs from last training step to extract losses for logging
        self._last_outputs = None
    
    def training_step(self, model, inputs):
        """Override training_step to capture outputs for logging."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss with outputs to get individual losses
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        # Store outputs for logging (detached to avoid memory issues)
        cls_loss = outputs.get("classification_loss")
        reg_loss = outputs.get("regression_loss")
        self._last_outputs = {
            "classification_loss": cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            "regression_loss": reg_loss.detach() if isinstance(reg_loss, torch.Tensor) else reg_loss,
        }
        
        # Handle multi-GPU
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Backward pass
        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach()
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log to add individual losses to logs."""
        # Add individual losses from last outputs if available
        if self._last_outputs is not None:
            if "classification_loss" in self._last_outputs:
                cls_loss = self._last_outputs["classification_loss"]
                if isinstance(cls_loss, torch.Tensor):
                    logs["classification_loss"] = cls_loss.item()
                elif cls_loss is not None:
                    logs["classification_loss"] = float(cls_loss)
            
            if "regression_loss" in self._last_outputs:
                reg_loss = self._last_outputs["regression_loss"]
                if isinstance(reg_loss, torch.Tensor):
                    logs["regression_loss"] = reg_loss.item()
                elif reg_loss is not None:
                    logs["regression_loss"] = float(reg_loss)
            
            # Clear after logging
            self._last_outputs = None
        
        # Call parent log method
        super().log(logs)
    
    def compute_loss(
        self,
        model: StemGainModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,  # Accept additional kwargs like num_items_in_batch
    ):
        """
        Compute multi-task loss.
        
        Args:
            model: The model to compute loss for
            inputs: Batch dictionary with audio, stem_label, gain_label
            return_outputs: Whether to return model outputs
        
        Returns:
            loss: Combined loss tensor
            outputs: (optional) Model outputs if return_outputs=True
        """
        audio = inputs["audio"]
        stem_labels = inputs["stem_label"]
        gain_labels = inputs["gain_label"]
        
        # Forward pass
        outputs = model(
            audio=audio,
            labels=stem_labels,
            gain_labels=gain_labels,
        )
        
        # Extract losses
        cls_loss = outputs.get("classification_loss", torch.tensor(0.0, device=audio.device))
        reg_loss = outputs.get("regression_loss", torch.tensor(0.0, device=audio.device))
        
        # Combined loss
        loss = (
            self.classification_weight * cls_loss +
            self.regression_weight * reg_loss
        )
        
        # Add individual losses to outputs for logging (as detached scalars)
        # The Trainer will automatically log these if they're in the outputs
        outputs["classification_loss"] = cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else cls_loss
        outputs["regression_loss"] = reg_loss.detach() if isinstance(reg_loss, torch.Tensor) else reg_loss
        outputs["loss"] = loss.detach() if isinstance(loss, torch.Tensor) else loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(
        self,
        model: StemGainModel,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Perform a prediction step for evaluation.
        
        Returns:
            loss: Combined loss (or None if labels are missing)
            logits: Tuple(classification_logits, gain_predictions)
            labels: Tuple(stem_labels, gain_labels) or None
        """
        has_labels = "stem_label" in inputs and "gain_label" in inputs
        
        # Forward pass
        audio = inputs["audio"]
        outputs = model(audio=audio)
        
        # Extract predictions
        classification_logits = outputs["classification_logits"]  # [batch, num_classes]
        gain_predictions = outputs["gain_prediction"].squeeze(-1)  # [batch]
        
        # Compute loss if labels available
        loss = None
        stem_labels = None
        gain_labels = None
        if has_labels:
            stem_labels = inputs["stem_label"]
            gain_labels = inputs["gain_label"]
            
            # Classification loss
            cls_loss = nn.functional.cross_entropy(classification_logits, stem_labels)
            
            # Regression loss
            reg_loss = nn.functional.mse_loss(gain_predictions, gain_labels)
            
            # Combined loss
            loss = (
                self.classification_weight * cls_loss +
                self.regression_weight * reg_loss
            )
        
        # Pack logits and labels as tuples so Trainer/EvalPrediction can handle them
        logits = (classification_logits, gain_predictions)
        labels = None
        if has_labels:
            labels = (stem_labels, gain_labels)
        
        return (loss, logits, labels)

