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
        
        # Average log probs for each sequence to avoid length bias
        # This normalizes for different sequence lengths between chosen/rejected
        num_tokens = loss_mask.sum(-1).float()
        num_tokens = torch.clamp(num_tokens, min=1.0)  # Avoid division by zero
        sequence_logps = per_token_logps.sum(-1) / num_tokens
        
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

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Override training_step to compute and store DPO-specific metrics for logging.
        
        Args:
            model: The model being trained
            inputs: Batch of training inputs
            num_items_in_batch: Optional number of items in batch (for newer transformers versions)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Backward pass through accelerator
        self.accelerator.backward(loss)

        # Store DPO metrics for logging (will be picked up by log() method)
        if not hasattr(self, "_stored_metrics"):
            self._stored_metrics = {}
        
        for key, value in metrics.items():
            if key != "loss":  # Loss is already handled by Trainer
                if isinstance(value, torch.Tensor):
                    self._stored_metrics[f"train/{key}"] = value.detach().cpu().item()
                else:
                    self._stored_metrics[f"train/{key}"] = value

        return loss.detach() / self.args.gradient_accumulation_steps

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:
        """
        Override log to include stored DPO metrics.
        This respects the logging_steps configuration from TrainingArguments.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for computing training speed
            **kwargs: Additional arguments passed by Trainer
        """
        # Add any stored metrics to the logs
        if hasattr(self, "_stored_metrics") and self._stored_metrics:
            logs.update(self._stored_metrics)
            self._stored_metrics = {}
        
        # Call parent's log method which handles the actual logging
        super().log(logs, start_time, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Override prediction_step to handle DPO-specific inputs and store metrics.
        
        Returns metrics that will be aggregated in compute_metrics().
        """
        with torch.no_grad():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Store metrics for aggregation in compute_metrics()
        # Initialize metrics storage if it doesn't exist or if it was cleared
        if not hasattr(self, "_eval_metrics") or not self._eval_metrics["losses"]:
            # Reset metrics at the start of a new evaluation run
            self._eval_metrics = {
                "losses": [],
                "chosen_rewards": [],
                "rejected_rewards": [],
                "reward_accuracies": [],
                "reward_margins": [],
            }
        
        # Store metrics (convert tensors to Python scalars for aggregation)
        self._eval_metrics["losses"].append(loss.detach().cpu().item())
        self._eval_metrics["chosen_rewards"].append(metrics["chosen_rewards"].detach().cpu().item())
        self._eval_metrics["rejected_rewards"].append(metrics["rejected_rewards"].detach().cpu().item())
        self._eval_metrics["reward_accuracies"].append(metrics["reward_accuracies"].detach().cpu().item())
        self._eval_metrics["reward_margins"].append(metrics["reward_margins"].detach().cpu().item())
        
        # Return dummy predictions/labels (not used for DPO, but required by Trainer)
        # The actual metrics are stored in _eval_metrics and aggregated in compute_metrics()
        batch_size = inputs["chosen_input_ids"].shape[0]
        dummy_predictions = torch.zeros(batch_size, 1)  # Dummy tensor
        dummy_labels = torch.zeros(batch_size, 1)  # Dummy tensor
        
        return (loss, dummy_predictions, dummy_labels)
    
    def compute_metrics(self, eval_pred):
        """
        Compute and aggregate DPO evaluation metrics.
        
        Args:
            eval_pred: EvalPrediction object containing predictions and labels
                      (not used for DPO, metrics are stored in _eval_metrics)
        
        Returns:
            Dictionary of aggregated metrics with "eval/" prefix
        """
        if not hasattr(self, "_eval_metrics") or not self._eval_metrics["losses"]:
            logger.warning("No evaluation metrics found. Returning empty metrics.")
            return {}
        
        # Aggregate metrics across all batches
        metrics = {
            "eval/loss": sum(self._eval_metrics["losses"]) / len(self._eval_metrics["losses"]),
            "eval/chosen_rewards": sum(self._eval_metrics["chosen_rewards"]) / len(self._eval_metrics["chosen_rewards"]),
            "eval/rejected_rewards": sum(self._eval_metrics["rejected_rewards"]) / len(self._eval_metrics["rejected_rewards"]),
            "eval/reward_accuracies": sum(self._eval_metrics["reward_accuracies"]) / len(self._eval_metrics["reward_accuracies"]),
            "eval/reward_margins": sum(self._eval_metrics["reward_margins"]) / len(self._eval_metrics["reward_margins"]),
        }
        
        # Clear stored metrics for next evaluation
        self._eval_metrics = {
            "losses": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
            "reward_accuracies": [],
            "reward_margins": [],
        }
        
        return metrics

