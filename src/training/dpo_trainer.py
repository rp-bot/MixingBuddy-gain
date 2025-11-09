"""
DPO (Direct Preference Optimization) Trainer for multimodal models.

Implements the DPO loss from Rafailov et al. (2023):
https://arxiv.org/abs/2305.18290
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from src.models.modular_multimodal_model import ModularMultimodalModel

logger = logging.getLogger(__name__)


class DPOTrainer(Trainer):
    """
    Trainer for Direct Preference Optimization with multimodal models.
    
    DPO directly optimizes the model to prefer chosen responses over rejected
    responses without requiring a separate reward model.
    """

    def __init__(
        self,
        model: ModularMultimodalModel,
        ref_model: Optional[ModularMultimodalModel] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        **kwargs,
    ):
        """
        Args:
            model: The model to train
            ref_model: Reference model for computing log probabilities (if None, uses frozen copy of model)
            beta: Temperature parameter controlling the strength of the preference optimization
            label_smoothing: Label smoothing factor for the DPO loss
            loss_type: Type of DPO loss ("sigmoid", "hinge", or "ipo")
            **kwargs: Additional arguments passed to Trainer
        """
        super().__init__(model=model, **kwargs)
        
        self.ref_model = ref_model
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        
        # If no reference model provided, create a frozen copy
        if self.ref_model is None:
            logger.info("No reference model provided. Creating frozen copy of the model.")
            self.ref_model = self._create_reference_model(model)
        
        # Ensure reference model is in eval mode and gradients are disabled
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _create_reference_model(
        self, model: ModularMultimodalModel
    ) -> ModularMultimodalModel:
        """Create a frozen copy of the model to use as reference."""
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def compute_loss(
        self,
        model: ModularMultimodalModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the DPO loss.
        
        Args:
            model: The model being trained
            inputs: Batch of inputs containing:
                - audio: Audio tensor (shared for chosen and rejected)
                - chosen_input_ids, chosen_attention_mask, chosen_labels
                - rejected_input_ids, rejected_attention_mask, rejected_labels
            return_outputs: Whether to return additional outputs
            num_items_in_batch: Optional number of items in batch (for compatibility with newer transformers)
            
        Returns:
            Loss tensor (and optionally a dict of additional outputs)
        """
        # Extract shared audio and separate text inputs
        audio = inputs["audio"]
        
        chosen_inputs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"],
            "labels": inputs["chosen_labels"],
            "audio": audio,
        }
        
        rejected_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"],
            "labels": inputs["rejected_labels"],
            "audio": audio,
        }
        
        # Compute log probabilities for chosen and rejected with the policy model
        policy_chosen_logps = self._get_batch_logps(model, chosen_inputs)
        policy_rejected_logps = self._get_batch_logps(model, rejected_inputs)
        
        # Compute log probabilities with the reference model
        with torch.no_grad():
            reference_chosen_logps = self._get_batch_logps(self.ref_model, chosen_inputs)
            reference_rejected_logps = self._get_batch_logps(self.ref_model, rejected_inputs)
        
        # Compute DPO loss
        loss, chosen_rewards, rejected_rewards = self._dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        # Compute metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        reward_margins = (chosen_rewards - rejected_rewards).mean()
        
        if return_outputs:
            outputs = {
                "loss": loss,
                "chosen_rewards": chosen_rewards.mean(),
                "rejected_rewards": rejected_rewards.mean(),
                "reward_accuracies": reward_accuracies,
                "reward_margins": reward_margins,
            }
            return loss, outputs
        
        return loss

    def _get_batch_logps(
        self,
        model: ModularMultimodalModel,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log probabilities of the labels under the model.
        
        Args:
            model: The model to evaluate
            inputs: Batch containing input_ids, attention_mask, labels, and audio
            
        Returns:
            Tensor of log probabilities for each sequence in the batch
        """
        # Forward pass through the model
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            audio=inputs["audio"],
        )
        
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get vocabulary size
        vocab_size = shift_logits.shape[-1]
        
        # Mask out ignored tokens (label = -100)
        loss_mask = (shift_labels != -100)
        
        # Replace -100 with 0 for gathering (we'll mask out the result anyway)
        # Clamp all labels to valid range [0, vocab_size) to avoid index out of bounds
        shift_labels_safe = shift_labels.clone()
        shift_labels_safe[shift_labels == -100] = 0  # Replace -100 with 0
        shift_labels_safe = shift_labels_safe.clamp(min=0, max=vocab_size - 1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of the labels
        per_token_logps = torch.gather(
            log_probs, dim=2, index=shift_labels_safe.unsqueeze(2)
        ).squeeze(2)
        
        # Mask out ignored tokens (set to 0 for -100 labels)
        per_token_logps = per_token_logps * loss_mask.float()
        
        # Sum log probs for each sequence (average over non-masked tokens)
        # Avoid division by zero
        loss_mask_sum = loss_mask.sum(-1).float()
        loss_mask_sum = torch.clamp(loss_mask_sum, min=1.0)  # At least 1 to avoid division by zero
        sequence_logps = per_token_logps.sum(-1) / loss_mask_sum
        
        return sequence_logps

    def _dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss given log probabilities.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy model
            policy_rejected_logps: Log probs of rejected responses under policy model
            reference_chosen_logps: Log probs of chosen responses under reference model
            reference_rejected_logps: Log probs of rejected responses under reference model
            
        Returns:
            Tuple of (loss, chosen_rewards, rejected_rewards)
        """
        # Compute implicit rewards
        policy_chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        policy_rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # Compute logits for the preference model
        logits = policy_chosen_rewards - policy_rejected_rewards
        
        # Compute loss based on loss type
        if self.loss_type == "sigmoid":
            # Standard DPO loss with sigmoid
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - logits)
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * (
                -F.logsigmoid(-logits)
            )
        
        loss = losses.mean()
        
        return loss, policy_chosen_rewards, policy_rejected_rewards

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle DPO-specific inputs.
        """
        with torch.no_grad():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # For evaluation, we care about the metrics
        return (loss, None, None)

