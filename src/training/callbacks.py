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
        for name, param in self.model.audio_projection.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_values.append(param.grad.abs().mean().item())

        avg_grad = sum(grad_values) / len(grad_values) if grad_values else 0.0
        self.grad_history.append(avg_grad)

        if current_step == 1 or current_step % 5 == 0:
            logger.info(
                "Projection gradients present=%s avg_magnitude=%.8f at step %d",
                has_grad,
                avg_grad,
                current_step,
            )
            if len(self.grad_history) > 1:
                initial_grad = self.grad_history[0]
                if initial_grad > 0:
                    improvement = (avg_grad / initial_grad) * 100
                    logger.debug("Gradient magnitude vs step 1: %.1f%%", improvement)
            if has_grad and avg_grad < 1e-6:
                logger.warning("Gradients are extremely small on projection layer")
