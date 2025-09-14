"""
Tests for training functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.training.trainer import (
    LoRATrainer,
    ExperimentTrackingCallback,
    compute_metrics,
    setup_optimizer_and_scheduler,
)


class TestLoRATrainer:
    """Test cases for LoRATrainer class."""

    def test_init(self, sample_config, mock_peft_model, mock_experiment_tracker):
        """Test LoRATrainer initialization."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config, mock_experiment_tracker)

        assert trainer.model == model
        assert trainer.config == sample_config
        assert trainer.experiment_tracker == mock_experiment_tracker
        assert trainer.trainer is None

    def test_init_without_experiment_tracker(self, sample_config, mock_peft_model):
        """Test LoRATrainer initialization without experiment tracker."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)

        assert trainer.model == model
        assert trainer.config == sample_config
        assert trainer.experiment_tracker is None

    def test_setup_training_args(self, sample_config, mock_peft_model):
        """Test setting up training arguments."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert (
            args.num_train_epochs
            == sample_config.training.training_args.num_train_epochs
        )
        assert (
            args.per_device_train_batch_size
            == sample_config.training.training_args.per_device_train_batch_size
        )
        assert args.learning_rate == sample_config.training.training_args.learning_rate
        assert args.seed == sample_config.env.seed

    def test_setup_training_args_with_run_name(self, sample_config, mock_peft_model):
        """Test setting up training arguments with custom run name."""
        sample_config.training.training_args.run_name = "custom-run-name"

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.run_name == "custom-run-name"

    def test_setup_training_args_generate_run_name(
        self, sample_config, mock_peft_model
    ):
        """Test generating run name when not provided."""
        # Remove run_name from config
        del sample_config.training.training_args.run_name

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        # Should generate a run name
        assert args.run_name is not None
        assert "lora-" in args.run_name

    def test_setup_data_collator(self, sample_config, mock_peft_model, mock_tokenizer):
        """Test setting up data collator."""
        model = Mock()
        model.get_model.return_value = mock_peft_model
        model.get_tokenizer.return_value = mock_tokenizer

        trainer = LoRATrainer(model, sample_config)
        collator = trainer.setup_data_collator()

        assert collator is not None

    def test_setup_callbacks_without_early_stopping(
        self, sample_config, mock_peft_model
    ):
        """Test setting up callbacks without early stopping."""
        sample_config.training.early_stopping.enabled = False

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        callbacks = trainer.setup_callbacks()

        assert len(callbacks) == 0

    def test_setup_callbacks_with_early_stopping(self, sample_config, mock_peft_model):
        """Test setting up callbacks with early stopping."""
        sample_config.training.early_stopping.enabled = True

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        callbacks = trainer.setup_callbacks()

        assert len(callbacks) == 1
        assert hasattr(callbacks[0], "early_stopping_patience")

    def test_setup_callbacks_with_experiment_tracker(
        self, sample_config, mock_peft_model, mock_experiment_tracker
    ):
        """Test setting up callbacks with experiment tracker."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config, mock_experiment_tracker)
        callbacks = trainer.setup_callbacks()

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ExperimentTrackingCallback)

    @patch("src.training.trainer.Trainer")
    def test_train(
        self,
        mock_trainer_class,
        sample_config,
        mock_peft_model,
        mock_tokenizer,
        mock_trainer,
    ):
        """Test training the model."""
        # Setup mocks
        mock_trainer_class.return_value = mock_trainer
        mock_dataset = Mock()
        mock_dataloader = Mock()
        mock_dataloader.dataset = mock_dataset

        model = Mock()
        model.get_model.return_value = mock_peft_model
        model.get_tokenizer.return_value = mock_tokenizer

        trainer = LoRATrainer(model, sample_config)
        result = trainer.train(mock_dataloader)

        assert result == mock_trainer
        assert trainer.trainer == mock_trainer
        mock_trainer.train.assert_called_once()

    @patch("src.training.trainer.Trainer")
    def test_train_with_eval_dataloader(
        self,
        mock_trainer_class,
        sample_config,
        mock_peft_model,
        mock_tokenizer,
        mock_trainer,
    ):
        """Test training with evaluation dataloader."""
        # Setup mocks
        mock_trainer_class.return_value = mock_trainer
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_train_dataloader = Mock()
        mock_train_dataloader.dataset = mock_train_dataset
        mock_eval_dataloader = Mock()
        mock_eval_dataloader.dataset = mock_eval_dataset

        model = Mock()
        model.get_model.return_value = mock_peft_model
        model.get_tokenizer.return_value = mock_tokenizer

        trainer = LoRATrainer(model, sample_config)
        result = trainer.train(mock_train_dataloader, mock_eval_dataloader)

        assert result == mock_trainer
        mock_trainer.train.assert_called_once()

    @patch("src.training.trainer.Trainer")
    def test_train_with_experiment_tracker(
        self,
        mock_trainer_class,
        sample_config,
        mock_peft_model,
        mock_tokenizer,
        mock_trainer,
        mock_experiment_tracker,
    ):
        """Test training with experiment tracker."""
        # Setup mocks
        mock_trainer_class.return_value = mock_trainer
        mock_dataset = Mock()
        mock_dataloader = Mock()
        mock_dataloader.dataset = mock_dataset

        model = Mock()
        model.get_model.return_value = mock_peft_model
        model.get_tokenizer.return_value = mock_tokenizer

        trainer = LoRATrainer(model, sample_config, mock_experiment_tracker)
        result = trainer.train(mock_dataloader)

        # Check that experiment tracker methods were called
        mock_experiment_tracker.log_params.assert_called_once()
        mock_experiment_tracker.log_metrics.assert_called_once()

    def test_evaluate(self, sample_config, mock_peft_model, mock_trainer):
        """Test evaluating the model."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        trainer.trainer = mock_trainer

        mock_dataset = Mock()
        mock_dataloader = Mock()
        mock_dataloader.dataset = mock_dataset

        result = trainer.evaluate(mock_dataloader)

        assert result == mock_trainer.evaluate.return_value
        mock_trainer.evaluate.assert_called_once()

    def test_evaluate_no_trainer(self, sample_config, mock_peft_model):
        """Test evaluating without initialized trainer."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)

        mock_dataloader = Mock()

        with pytest.raises(ValueError, match="Trainer not initialized"):
            trainer.evaluate(mock_dataloader)

    def test_save_model(self, sample_config, mock_peft_model, temp_dir):
        """Test saving the model."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        trainer.trainer = Mock()

        output_dir = temp_dir / "saved_model"
        trainer.save_model(output_dir)

        model.save_model.assert_called_once_with(output_dir)
        trainer.trainer.save_state.assert_called_once()

    def test_save_model_default_path(self, sample_config, mock_peft_model):
        """Test saving the model with default path."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        trainer.trainer = Mock()

        trainer.save_model()

        # Should use default path
        model.save_model.assert_called_once()

    def test_save_model_with_experiment_tracker(
        self, sample_config, mock_peft_model, mock_experiment_tracker, temp_dir
    ):
        """Test saving the model with experiment tracker."""
        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config, mock_experiment_tracker)
        trainer.trainer = Mock()

        output_dir = temp_dir / "saved_model"
        trainer.save_model(output_dir)

        model.save_model.assert_called_once_with(output_dir)
        mock_experiment_tracker.log_artifacts.assert_called_once()

    def test_get_trainable_parameters_info(self, sample_config, mock_peft_model):
        """Test getting trainable parameters info."""
        model = Mock()
        model.get_model.return_value = mock_peft_model
        model.get_trainable_parameters.return_value = {
            "total": 1000,
            "trainable": 100,
            "percentage": 10.0,
        }

        trainer = LoRATrainer(model, sample_config)
        info = trainer.get_trainable_parameters_info()

        assert info == {"total": 1000, "trainable": 100, "percentage": 10.0}
        model.get_trainable_parameters.assert_called_once()


class TestExperimentTrackingCallback:
    """Test cases for ExperimentTrackingCallback class."""

    def test_init(self, mock_experiment_tracker):
        """Test ExperimentTrackingCallback initialization."""
        callback = ExperimentTrackingCallback(mock_experiment_tracker)

        assert callback.experiment_tracker == mock_experiment_tracker
        assert callback.step == 0

    def test_on_log(self, mock_experiment_tracker):
        """Test on_log callback."""
        callback = ExperimentTrackingCallback(mock_experiment_tracker)

        # Mock state and logs
        state = Mock()
        state.global_step = 100
        logs = {"loss": 0.5, "accuracy": 0.8}

        callback.on_log(None, state, None, logs=logs)

        mock_experiment_tracker.log_metrics.assert_called_once_with(logs, step=100)

    def test_on_log_no_logs(self, mock_experiment_tracker):
        """Test on_log callback with no logs."""
        callback = ExperimentTrackingCallback(mock_experiment_tracker)

        state = Mock()
        state.global_step = 100

        callback.on_log(None, state, None, logs=None)

        mock_experiment_tracker.log_metrics.assert_not_called()

    def test_on_log_no_experiment_tracker(self):
        """Test on_log callback with no experiment tracker."""
        callback = ExperimentTrackingCallback(None)

        state = Mock()
        state.global_step = 100
        logs = {"loss": 0.5}

        # Should not raise an error
        callback.on_log(None, state, None, logs=logs)

    def test_on_save(self, mock_experiment_tracker, temp_dir):
        """Test on_save callback."""
        callback = ExperimentTrackingCallback(mock_experiment_tracker)

        # Mock args and state
        args = Mock()
        args.output_dir = str(temp_dir)
        state = Mock()
        state.global_step = 100

        # Create checkpoint directory
        checkpoint_dir = temp_dir / "checkpoint-100"
        checkpoint_dir.mkdir()

        callback.on_save(args, state, None)

        mock_experiment_tracker.log_artifacts.assert_called_once()

    def test_on_save_no_checkpoint_dir(self, mock_experiment_tracker, temp_dir):
        """Test on_save callback when checkpoint directory doesn't exist."""
        callback = ExperimentTrackingCallback(mock_experiment_tracker)

        # Mock args and state
        args = Mock()
        args.output_dir = str(temp_dir)
        state = Mock()
        state.global_step = 100

        # Don't create checkpoint directory
        callback.on_save(args, state, None)

        mock_experiment_tracker.log_artifacts.assert_not_called()

    def test_on_save_no_experiment_tracker(self, temp_dir):
        """Test on_save callback with no experiment tracker."""
        callback = ExperimentTrackingCallback(None)

        # Mock args and state
        args = Mock()
        args.output_dir = str(temp_dir)
        state = Mock()
        state.global_step = 100

        # Should not raise an error
        callback.on_save(args, state, None)


class TestComputeMetrics:
    """Test cases for compute_metrics function."""

    def test_compute_metrics(self):
        """Test computing metrics for evaluation."""
        # Mock predictions and labels
        predictions = torch.randn(2, 10, 1000)  # (batch_size, seq_len, vocab_size)
        labels = torch.randint(0, 1000, (2, 10))

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        assert "perplexity" in metrics
        assert "loss" in metrics
        assert isinstance(metrics["perplexity"], float)
        assert isinstance(metrics["loss"], float)
        assert metrics["perplexity"] > 0
        assert metrics["loss"] > 0

    def test_compute_metrics_single_sample(self):
        """Test computing metrics for single sample."""
        predictions = torch.randn(1, 5, 1000)
        labels = torch.randint(0, 1000, (1, 5))

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        assert "perplexity" in metrics
        assert "loss" in metrics

    def test_compute_metrics_perfect_prediction(self):
        """Test computing metrics with perfect prediction."""
        # Create predictions that match labels exactly
        vocab_size = 1000
        batch_size = 2
        seq_len = 5

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create predictions with high probability for correct tokens
        predictions = torch.zeros(batch_size, seq_len, vocab_size)
        for i in range(batch_size):
            for j in range(seq_len):
                predictions[i, j, labels[i, j]] = 10.0  # High logit for correct token

        eval_pred = (predictions, labels)
        metrics = compute_metrics(eval_pred)

        assert "perplexity" in metrics
        assert "loss" in metrics
        # Should have low loss and perplexity for perfect predictions
        assert metrics["loss"] < 1.0
        assert metrics["perplexity"] < 3.0


class TestSetupOptimizerAndScheduler:
    """Test cases for setup_optimizer_and_scheduler function."""

    def test_setup_optimizer_and_scheduler(self, sample_config, mock_peft_model):
        """Test setting up optimizer and scheduler."""
        # Mock model parameters
        param1 = torch.randn(10, 10, requires_grad=True)
        param2 = torch.randn(5, 5, requires_grad=False)
        mock_peft_model.parameters.return_value = [param1, param2]

        num_training_steps = 1000

        optimizer, scheduler = setup_optimizer_and_scheduler(
            mock_peft_model, sample_config, num_training_steps
        )

        assert optimizer is not None
        assert scheduler is not None

        # Check optimizer parameters
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]["params"]) == 1  # Only trainable params

        # Check scheduler
        assert hasattr(scheduler, "get_last_lr")

    def test_setup_optimizer_and_scheduler_no_trainable_params(
        self, sample_config, mock_peft_model
    ):
        """Test setting up optimizer and scheduler with no trainable parameters."""
        # Mock model with no trainable parameters
        param = torch.randn(10, 10, requires_grad=False)
        mock_peft_model.parameters.return_value = [param]

        num_training_steps = 1000

        optimizer, scheduler = setup_optimizer_and_scheduler(
            mock_peft_model, sample_config, num_training_steps
        )

        assert optimizer is not None
        assert scheduler is not None

        # Should have empty param groups
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]["params"]) == 0


class TestTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_trainer_with_mixed_precision_fp16(self, sample_config, mock_peft_model):
        """Test trainer with FP16 mixed precision."""
        sample_config.env.mixed_precision = "fp16"

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.fp16 is True
        assert args.bf16 is False

    def test_trainer_with_mixed_precision_bf16(self, sample_config, mock_peft_model):
        """Test trainer with BF16 mixed precision."""
        sample_config.env.mixed_precision = "bf16"

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.fp16 is False
        assert args.bf16 is True

    def test_trainer_with_gradient_checkpointing(self, sample_config, mock_peft_model):
        """Test trainer with gradient checkpointing enabled."""
        sample_config.model.training.gradient_checkpointing = True

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.gradient_checkpointing is True

    def test_trainer_with_custom_optimizer_params(self, sample_config, mock_peft_model):
        """Test trainer with custom optimizer parameters."""
        sample_config.training.optimizer.adam_beta1 = 0.95
        sample_config.training.optimizer.adam_beta2 = 0.999
        sample_config.training.optimizer.adam_epsilon = 1e-6

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.adam_beta1 == 0.95
        assert args.adam_beta2 == 0.999
        assert args.adam_epsilon == 1e-6

    def test_trainer_with_custom_gradient_clipping(
        self, sample_config, mock_peft_model
    ):
        """Test trainer with custom gradient clipping."""
        sample_config.training.gradient_clipping.max_grad_norm = 0.5

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.max_grad_norm == 0.5

    def test_trainer_with_custom_dataloader_params(
        self, sample_config, mock_peft_model
    ):
        """Test trainer with custom dataloader parameters."""
        sample_config.data.dataloader.num_workers = 4

        model = Mock()
        model.get_model.return_value = mock_peft_model

        trainer = LoRATrainer(model, sample_config)
        args = trainer.setup_training_args()

        assert args.dataloader_num_workers == 4
