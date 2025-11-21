"""
Custom trainer for multi-label stem classification.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import Trainer

from src.models.stem_gain_model import StemGainModel

logger = logging.getLogger(__name__)


class StemGainTrainer(Trainer):
    """Trainer for multi-label stem classification.
    
    Implements binary cross-entropy loss for multi-label classification.
    """
    
    def __init__(
        self,
        model: StemGainModel,
        **kwargs,
    ):
        """
        Args:
            model: StemGainModel instance
            **kwargs: Additional arguments passed to Trainer
        """
        super().__init__(model=model, **kwargs)
        # Store outputs from last training step to extract losses for logging
        self._last_outputs = None
        # Accumulate classification_loss across gradient accumulation steps
        self._accumulated_cls_loss = 0.0
        self._accumulation_steps = 0
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to capture outputs for logging."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss with outputs to get individual losses
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        # Accumulate classification_loss across gradient accumulation steps
        cls_loss = outputs.get("classification_loss")
        if isinstance(cls_loss, torch.Tensor):
            cls_loss_value = cls_loss.detach().item()
        else:
            cls_loss_value = float(cls_loss) if cls_loss is not None else 0.0
        
        self._accumulated_cls_loss += cls_loss_value
        self._accumulation_steps += 1
        
        # Store outputs for logging (detached to avoid memory issues)
        self._last_outputs = {
            "classification_loss": cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else cls_loss,
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
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log to add individual losses to logs."""
        # Add accumulated classification_loss (averaged across gradient accumulation steps)
        # This matches how the parent Trainer handles the main 'loss' field
        if self._accumulation_steps > 0:
            avg_cls_loss = self._accumulated_cls_loss / self._accumulation_steps
            logs["classification_loss"] = avg_cls_loss
            # Reset accumulation for next logging cycle
            self._accumulated_cls_loss = 0.0
            self._accumulation_steps = 0
        
        # Clear last outputs after logging
        self._last_outputs = None
        
        # Call parent log method
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
    
    def compute_loss(
        self,
        model: StemGainModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,  # Accept additional kwargs like num_items_in_batch
    ):
        """
        Compute multi-label classification loss.
        
        Args:
            model: The model to compute loss for
            inputs: Batch dictionary with audio, multi_label
            return_outputs: Whether to return model outputs
        
        Returns:
            loss: Classification loss tensor
            outputs: (optional) Model outputs if return_outputs=True
        """
        audio = inputs["audio"]
        multi_labels = inputs["multi_label"]
        
        # Forward pass
        outputs = model(
            audio=audio,
            labels=multi_labels,
        )
        
        # Extract loss
        loss = outputs.get("loss", torch.tensor(0.0, device=audio.device))
        cls_loss = outputs.get("classification_loss", loss)
        
        # Add individual losses to outputs for logging (as detached scalars)
        outputs["classification_loss"] = cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else cls_loss
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
            loss: Classification loss (or None if labels are missing)
            logits: classification_logits [batch_size, num_classes]
            labels: multi_labels [batch_size, num_classes] or None
        """
        has_labels = "multi_label" in inputs

        # Ensure no gradients/activations are kept during evaluation to avoid OOM
        with torch.no_grad():
            # Forward pass
            audio = inputs["audio"]
            multi_labels = inputs.get("multi_label")
            
            outputs = model(audio=audio, labels=multi_labels if has_labels else None)
            
            # Extract predictions
            classification_logits = outputs["classification_logits"]  # [batch, num_classes]
            
            # Extract loss if labels available
            loss = outputs.get("loss") if has_labels else None
            
            # Pack logits and labels
            logits = classification_logits
            labels = multi_labels if has_labels else None
        
        return (loss, logits, labels)

