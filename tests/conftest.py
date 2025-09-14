"""
Pytest configuration and shared fixtures for testing.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.data.dataset import AutomaticMixingDataset, DataProcessor
from src.models.lora_model import LoRAModel
from src.training.trainer import LoRATrainer
from src.utils.experiment_tracking import ExperimentTracker


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    """Sample data for testing."""
    return [
        {
            "instruction": "Analyze the mixing parameters for track 1",
            "response": "Track 1 requires EQ adjustments at 2kHz, compression with ratio 3:1, and reverb with 1.2s decay time.",
        },
        {
            "instruction": "What are the optimal settings for vocal compression?",
            "response": "For vocals, use a 4:1 compression ratio with 3ms attack and 100ms release time.",
        },
        {
            "input": "Mix this drum track",
            "output": "Apply high-pass filter at 80Hz, compress with 2:1 ratio, and add reverb with 0.8s decay.",
        },
        {
            "text": "This is a plain text sample for testing the dataset functionality.",
        },
    ]


@pytest.fixture
def sample_config() -> DictConfig:
    """Sample configuration for testing."""
    config_dict = {
        "model": {
            "pretrained_model_name_or_path": "microsoft/DialoGPT-small",
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "inference_mode": False,
            },
            "training": {
                "gradient_checkpointing": False,
            },
        },
        "data": {
            "dataset": {
                "path": "data/processed",
                "train_file": "train.jsonl",
                "validation_file": "validation.jsonl",
                "test_file": "test.jsonl",
            },
            "processing": {
                "max_length": 512,
                "padding": "max_length",
                "truncation": True,
                "add_special_tokens": True,
            },
            "dataloader": {
                "batch_size": 2,
                "shuffle": True,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": True,
            },
        },
        "training": {
            "training_args": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "linear",
                "logging_steps": 10,
                "eval_steps": 100,
                "save_steps": 100,
                "save_total_limit": 3,
                "evaluation_strategy": "steps",
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "report_to": "none",
                "run_name": "test-run",
            },
            "optimizer": {
                "type": "adamw_torch",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
            },
            "gradient_clipping": {
                "max_grad_norm": 1.0,
            },
            "early_stopping": {
                "enabled": False,
                "patience": 3,
                "threshold": 0.001,
            },
        },
        "paths": {
            "output_dir": "outputs/test",
            "logs_dir": "logs/test",
        },
        "env": {
            "seed": 42,
            "mixed_precision": "fp16",
        },
        "experiment_tracking": {
            "project": "test-project",
            "name": "test-run",
            "tags": ["test"],
            "notes": "Test run",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock(spec=AutoTokenizer)
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 1
    tokenizer.bos_token = "<bos>"
    tokenizer.bos_token_id = 2

    # Mock tokenize method
    def mock_tokenize(text, **kwargs):
        # Simple tokenization simulation
        tokens = text.split()[: kwargs.get("max_length", 512)]
        input_ids = [hash(token) % 1000 + 10 for token in tokens]  # Mock token IDs
        attention_mask = [1] * len(input_ids)

        # Pad if needed
        if kwargs.get("padding") == "max_length":
            max_len = kwargs.get("max_length", 512)
            while len(input_ids) < max_len:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

    tokenizer.side_effect = mock_tokenize
    return tokenizer


@pytest.fixture
def temp_data_file(sample_data):
    """Create temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for sample in sample_data:
            f.write(json.dumps(sample) + "\n")
        temp_file = f.name

    yield temp_file

    # Cleanup
    Path(temp_file).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.train.return_value = None
    model.eval.return_value = None
    model.parameters.return_value = [torch.randn(10, 10)]
    model.named_parameters.return_value = [("test_param", torch.randn(10, 10))]
    model.generate.return_value = torch.randint(0, 1000, (1, 10))
    return model


@pytest.fixture
def mock_peft_model(mock_model):
    """Mock PEFT model for testing."""
    peft_model = Mock()
    peft_model.train.return_value = None
    peft_model.eval.return_value = None
    peft_model.parameters.return_value = [torch.randn(10, 10)]
    peft_model.named_parameters.return_value = [("lora_A", torch.randn(10, 10))]
    peft_model.generate.return_value = torch.randint(0, 1000, (1, 10))
    peft_model.print_trainable_parameters.return_value = None
    peft_model.save_pretrained.return_value = None
    peft_model.merge_and_unload.return_value = mock_model
    return peft_model


@pytest.fixture
def mock_experiment_tracker():
    """Mock experiment tracker for testing."""
    tracker = Mock(spec=ExperimentTracker)
    tracker.backend = "wandb"
    tracker.run = Mock()
    tracker.log_params.return_value = None
    tracker.log_metrics.return_value = None
    tracker.log_artifacts.return_value = None
    tracker.log_model.return_value = None
    tracker.watch_model.return_value = None
    tracker.finish.return_value = None
    tracker.get_run_id.return_value = "test-run-id"
    tracker.get_run_url.return_value = "https://wandb.ai/test/project/runs/test-run-id"
    return tracker


@pytest.fixture
def mock_trainer():
    """Mock trainer for testing."""
    trainer = Mock()
    trainer.train.return_value = Mock(
        training_loss=0.5,
        metrics={
            "train_runtime": 100.0,
            "train_samples_per_second": 10.0,
            "train_steps_per_second": 1.0,
        },
    )
    trainer.evaluate.return_value = {"eval_loss": 0.6, "eval_perplexity": 1.8}
    trainer.save_state.return_value = None
    return trainer


# Test data for metrics
@pytest.fixture
def sample_predictions():
    """Sample predictions for metrics testing."""
    return [
        "Track 1 requires EQ at 2kHz, compression 3:1 ratio, reverb 1.2s decay.",
        "Vocal compression: 4:1 ratio, 3ms attack, 100ms release.",
        "Drum mix: high-pass 80Hz, compress 2:1, reverb 0.8s decay.",
    ]


@pytest.fixture
def sample_references():
    """Sample references for metrics testing."""
    return [
        "Track 1 needs EQ adjustments at 2kHz, compression with 3:1 ratio, and reverb with 1.2s decay time.",
        "For vocals, use 4:1 compression ratio with 3ms attack and 100ms release time.",
        "Apply high-pass filter at 80Hz, compress with 2:1 ratio, and add reverb with 0.8s decay.",
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Skip tests that require external dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests that might be slow
        if "integration" in item.nodeid or "e2e" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark tests that require GPU
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
