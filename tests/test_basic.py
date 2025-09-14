"""
Basic tests to verify the testing setup is working correctly.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all main modules can be imported."""
    try:
        from src.data.dataset import AutomaticMixingDataset
        from src.evaluation.metrics import compute_automatic_mixing_metrics
        from src.models.lora_model import LoRAModel
        from src.training.trainer import LoRATrainer
        from src.utils.experiment_tracking import ExperimentTracker

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    from src.evaluation.metrics import extract_mixing_parameters

    # Test parameter extraction
    text = "Apply EQ at 2kHz with 3dB boost"
    params = extract_mixing_parameters(text)

    assert isinstance(params, dict)
    assert "eq_frequencies" in params
    assert 2.0 in params["eq_frequencies"]


def test_config_loading():
    """Test that configuration can be loaded."""
    from omegaconf import DictConfig, OmegaConf

    config_dict = {"test": True, "nested": {"value": 42}}

    config = OmegaConf.create(config_dict)
    assert isinstance(config, DictConfig)
    assert config.test is True
    assert config.nested.value == 42


def test_pytest_fixtures():
    """Test that pytest fixtures are working."""
    # This test uses fixtures from conftest.py
    assert True


@pytest.mark.unit
def test_unit_marker():
    """Test that unit test markers work."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow test markers work."""
    assert True


@pytest.mark.gpu
def test_gpu_marker():
    """Test that GPU test markers work."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration test markers work."""
    assert True
