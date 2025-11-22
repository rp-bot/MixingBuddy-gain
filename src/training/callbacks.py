import logging
from transformers import TrainerCallback


logger = logging.getLogger(__name__)


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
        if self.optimizer_checked:
            return
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            logger.info(
                "Optimizer not yet available in callback; will check on first step."
            )
            return

        proj_param_ids = {id(p) for p in self.model.audio_projection.parameters()}
        optimizer_param_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        proj_in_optimizer = proj_param_ids.issubset(optimizer_param_ids)
        logger.info("Projection parameters in optimizer: %s", proj_in_optimizer)
        logger.debug(
            "Projection param count=%d, optimizer param count=%d",
            len(proj_param_ids),
            len(optimizer_param_ids),
        )

        if not proj_in_optimizer:
            logger.warning(
                "Projection parameters NOT in optimizer; checking requires_grad status"
            )
            for name, param in self.model.audio_projection.named_parameters():
                logger.warning("%s: requires_grad=%s", name, param.requires_grad)

        self.optimizer_checked = True

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1

        if self.step == 1 and not self.optimizer_checked:
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                proj_param_ids = {
                    id(p) for p in self.model.audio_projection.parameters()
                }
                optimizer_param_ids = {
                    id(p) for group in optimizer.param_groups for p in group["params"]
                }
                proj_in_optimizer = proj_param_ids.issubset(optimizer_param_ids)
                logger.info(
                    "Projection parameters in optimizer (first step): %s",
                    proj_in_optimizer,
                )
                logger.debug(
                    "Projection param count=%d, optimizer param count=%d",
                    len(proj_param_ids),
                    len(optimizer_param_ids),
                )
                if not proj_in_optimizer:
                    logger.warning(
                        "Projection parameters NOT in optimizer at first step"
                    )
                    for name, param in self.model.audio_projection.named_parameters():
                        logger.warning(
                            "%s: requires_grad=%s", name, param.requires_grad
                        )
                self.optimizer_checked = True

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
            logger.debug(
                "Weight change avg since step 1 at step %d: %.8f", self.step, avg_change
            )

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Check gradients BEFORE optimizer applies and zeros them."""
        current_step = state.global_step + 1

        has_grad = False
        grad_values = []
        grad_max_values = []
        param_names_with_grad = []
        param_names_without_grad = []
        
        # Check all projection parameters
        for name, param in self.model.audio_projection.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_mean = param.grad.abs().mean().item()
                grad_max = param.grad.abs().max().item()
                grad_values.append(grad_mean)
                grad_max_values.append(grad_max)
                param_names_with_grad.append(name)
            else:
                param_names_without_grad.append(name)

        avg_grad = sum(grad_values) / len(grad_values) if grad_values else 0.0
        max_grad = max(grad_max_values) if grad_max_values else 0.0
        self.grad_history.append(avg_grad)

        if current_step == 1 or current_step % 5 == 0:
            # Use scientific notation for very small values to see actual magnitude
            if avg_grad > 0 and avg_grad < 1e-3:
                grad_str = f"{avg_grad:.2e}"
                max_grad_str = f"{max_grad:.2e}"
            else:
                grad_str = f"{avg_grad:.8f}"
                max_grad_str = f"{max_grad:.8f}"
            
            logger.info(
                "Projection gradients present=%s avg_magnitude=%s max_magnitude=%s at step %d",
                has_grad,
                grad_str,
                max_grad_str,
                current_step,
            )
            
            # Log which parameters have gradients (helpful for debugging)
            if has_grad and len(param_names_with_grad) > 0:
                logger.debug(
                    "Projection params WITH gradients (%d): %s", 
                    len(param_names_with_grad),
                    ", ".join(param_names_with_grad)
                )
            
            # Log which parameters don't have gradients (indicates a problem)
            if len(param_names_without_grad) > 0:
                logger.warning(
                    "Projection params WITHOUT gradients (%d): %s. "
                    "This may indicate gradient flow issues.",
                    len(param_names_without_grad),
                    ", ".join(param_names_without_grad)
                )
            
            if len(self.grad_history) > 1:
                initial_grad = self.grad_history[0]
                if initial_grad > 0:
                    improvement = (avg_grad / initial_grad) * 100
                    logger.debug("Gradient magnitude vs step 1: %.1f%%", improvement)
            
            if has_grad and avg_grad < 1e-6:
                logger.warning(
                    "Gradients are extremely small on projection layer (avg=%.2e, max=%.2e). "
                    "This is expected in autoregressive training since projection only gets gradients from prefill step. "
                    "Consider increasing auxiliary_loss_weight or using gradient scaling.",
                    avg_grad,
                    max_grad,
                )
            elif not has_grad:
                logger.error(
                    "No gradients found on projection layer parameters at step %d. "
                    "This indicates a gradient flow problem. Check: "
                    "1) projection layer requires_grad=True, "
                    "2) projection layer is in optimizer, "
                    "3) loss computation includes projection layer in computation graph.",
                    current_step,
                )


class ProjectionGradientScalingCallback(TrainerCallback):
    """Scale up projection layer gradients to improve training signal."""
    
    def __init__(self, model, scale_factor: float = 10.0):
        """
        Args:
            model: The model containing the projection layer
            scale_factor: Multiplier for projection layer gradients (default: 10.0)
        """
        self.model = model
        self.scale_factor = scale_factor
        logger.info(
            f"ProjectionGradientScalingCallback initialized with scale_factor={scale_factor}"
        )
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Scale projection layer gradients BEFORE optimizer step."""
        current_step = state.global_step + 1
        scaled_count = 0
        total_before_magnitude = 0.0
        total_after_magnitude = 0.0
        max_before = 0.0
        max_after = 0.0
        
        for name, param in self.model.audio_projection.named_parameters():
            if param.grad is not None:
                # Record magnitude before scaling
                before_mean = param.grad.abs().mean().item()
                before_max = param.grad.abs().max().item()
                total_before_magnitude += before_mean
                max_before = max(max_before, before_max)
                
                # Scale up the gradients
                param.grad.mul_(self.scale_factor)
                
                # Record magnitude after scaling
                after_mean = param.grad.abs().mean().item()
                after_max = param.grad.abs().max().item()
                total_after_magnitude += after_mean
                max_after = max(max_after, after_max)
                
                scaled_count += 1
        
        if scaled_count > 0 and (current_step == 1 or current_step % 5 == 0):
            avg_before = total_before_magnitude / scaled_count
            avg_after = total_after_magnitude / scaled_count
            improvement = (avg_after / avg_before) if avg_before > 0 else 0.0
            
            logger.info(
                f"Gradient scaling: {scaled_count} projection params scaled by {self.scale_factor}x "
                f"(before: avg={avg_before:.2e}, max={max_before:.2e} | "
                f"after: avg={avg_after:.2e}, max={max_after:.2e}, actual_scale={improvement:.1f}x)"
            )
