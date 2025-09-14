"""
Tests for experiment tracking functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.utils.experiment_tracking import (
    ExperimentTracker,
    setup_logging,
    save_experiment_config,
)


class TestExperimentTracker:
    """Test cases for ExperimentTracker class."""

    def test_init_wandb(self, sample_config):
        """Test initializing ExperimentTracker with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_run.name = "test-run"
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            assert tracker.backend == "wandb"
            assert tracker.run == mock_run
            mock_wandb.init.assert_called_once()

    def test_init_mlflow(self, sample_config):
        """Test initializing ExperimentTracker with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_run.info.run_id = "test-run-id"
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            assert tracker.backend == "mlflow"
            assert tracker.run == mock_run
            mock_mlflow.set_experiment.assert_called_once()
            mock_mlflow.start_run.assert_called_once()

    def test_init_unsupported_backend(self, sample_config):
        """Test initializing ExperimentTracker with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            ExperimentTracker(sample_config, backend="unsupported")

    def test_init_wandb_failure(self, sample_config):
        """Test handling WandB initialization failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_wandb.init.side_effect = Exception("WandB error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            assert tracker.backend == "wandb"
            assert tracker.run is None

    def test_init_mlflow_failure(self, sample_config):
        """Test handling MLflow initialization failure."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_mlflow.start_run.side_effect = Exception("MLflow error")

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            assert tracker.backend == "mlflow"
            assert tracker.run is None

    def test_flatten_config(self, sample_config):
        """Test flattening nested configuration."""
        tracker = ExperimentTracker(sample_config, backend="wandb")

        flattened = tracker._flatten_config(sample_config)

        assert isinstance(flattened, dict)
        assert "model.pretrained_model_name_or_path" in flattened
        assert "training.training_args.learning_rate" in flattened
        assert (
            flattened["model.pretrained_model_name_or_path"]
            == sample_config.model.pretrained_model_name_or_path
        )

    def test_log_params_wandb(self, sample_config):
        """Test logging parameters with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            params = {"param1": "value1", "param2": 42}

            tracker.log_params(params)

            mock_wandb.config.update.assert_called_once_with(params)

    def test_log_params_mlflow(self, sample_config):
        """Test logging parameters with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")
            params = {"param1": "value1", "param2": 42}

            tracker.log_params(params)

            assert mock_mlflow.log_param.call_count == 2
            mock_mlflow.log_param.assert_any_call("param1", "value1")
            mock_mlflow.log_param.assert_any_call("param2", 42)

    def test_log_params_no_run(self, sample_config):
        """Test logging parameters when no run is active."""
        tracker = ExperimentTracker(sample_config, backend="wandb")
        tracker.run = None

        params = {"param1": "value1"}

        # Should not raise an error
        tracker.log_params(params)

    def test_log_metrics_wandb(self, sample_config):
        """Test logging metrics with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            metrics = {"loss": 0.5, "accuracy": 0.8}

            tracker.log_metrics(metrics, step=100)

            mock_wandb.log.assert_called_once_with(metrics, step=100)

    def test_log_metrics_mlflow(self, sample_config):
        """Test logging metrics with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")
            metrics = {"loss": 0.5, "accuracy": 0.8}

            tracker.log_metrics(metrics, step=100)

            assert mock_mlflow.log_metric.call_count == 2
            mock_mlflow.log_metric.assert_any_call("loss", 0.5, step=100)
            mock_mlflow.log_metric.assert_any_call("accuracy", 0.8, step=100)

    def test_log_metrics_no_step(self, sample_config):
        """Test logging metrics without step."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            metrics = {"loss": 0.5}

            tracker.log_metrics(metrics)

            mock_wandb.log.assert_called_once_with(metrics, step=None)

    def test_log_artifacts_wandb(self, sample_config, temp_dir):
        """Test logging artifacts with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            tracker.log_artifacts(str(temp_dir), "test_artifacts")

            mock_wandb.log_artifact.assert_called_once_with(
                str(temp_dir), name="test_artifacts"
            )

    def test_log_artifacts_mlflow(self, sample_config, temp_dir):
        """Test logging artifacts with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            tracker.log_artifacts(str(temp_dir), "test_artifacts")

            mock_mlflow.log_artifacts.assert_called_once_with(
                str(temp_dir), "test_artifacts"
            )

    def test_log_artifacts_no_artifact_path(self, sample_config, temp_dir):
        """Test logging artifacts without artifact path."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            tracker.log_artifacts(str(temp_dir))

            mock_wandb.log_artifact.assert_called_once_with(
                str(temp_dir), name="artifacts"
            )

    def test_log_model_wandb(self, sample_config, temp_dir):
        """Test logging model with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            model_path = str(temp_dir / "model")

            tracker.log_model(model_path, "test_model")

            mock_wandb.log_model.assert_called_once_with(model_path, name="test_model")

    def test_log_model_mlflow(self, sample_config, temp_dir):
        """Test logging model with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")
            model_path = str(temp_dir / "model")

            tracker.log_model(model_path, "test_model")

            mock_mlflow.log_model.assert_called_once_with(model_path, "test_model")

    def test_log_plots_wandb(self, sample_config):
        """Test logging plots with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            plots = {"plot1": Mock(), "plot2": Mock()}

            tracker.log_plots(plots)

            mock_wandb.log.assert_called_once_with(plots)

    def test_log_plots_mlflow(self, sample_config):
        """Test logging plots with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")
            plots = {"plot1": Mock(), "plot2": Mock()}

            tracker.log_plots(plots)

            assert mock_mlflow.log_figure.call_count == 2
            mock_mlflow.log_figure.assert_any_call(plots["plot1"], "plot1.png")
            mock_mlflow.log_figure.assert_any_call(plots["plot2"], "plot2.png")

    def test_watch_model_wandb(self, sample_config, mock_model):
        """Test watching model with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            tracker.watch_model(mock_model, log="gradients", log_freq=50)

            mock_wandb.watch.assert_called_once_with(
                mock_model, log="gradients", log_freq=50
            )

    def test_watch_model_mlflow(self, sample_config, mock_model):
        """Test watching model with MLflow backend (should be ignored)."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            # Should not raise an error
            tracker.watch_model(mock_model)

    def test_watch_model_no_run(self, sample_config, mock_model):
        """Test watching model when no run is active."""
        tracker = ExperimentTracker(sample_config, backend="wandb")
        tracker.run = None

        # Should not raise an error
        tracker.watch_model(mock_model)

    def test_finish_wandb(self, sample_config):
        """Test finishing WandB run."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")
            tracker.finish()

            mock_wandb.finish.assert_called_once()

    def test_finish_mlflow(self, sample_config):
        """Test finishing MLflow run."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")
            tracker.finish()

            mock_mlflow.end_run.assert_called_once()

    def test_finish_no_run(self, sample_config):
        """Test finishing when no run is active."""
        tracker = ExperimentTracker(sample_config, backend="wandb")
        tracker.run = None

        # Should not raise an error
        tracker.finish()

    def test_get_run_id_wandb(self, sample_config):
        """Test getting run ID with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_run.id = "wandb-run-id"
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            run_id = tracker.get_run_id()
            assert run_id == "wandb-run-id"

    def test_get_run_id_mlflow(self, sample_config):
        """Test getting run ID with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_run.info.run_id = "mlflow-run-id"
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            run_id = tracker.get_run_id()
            assert run_id == "mlflow-run-id"

    def test_get_run_id_no_run(self, sample_config):
        """Test getting run ID when no run is active."""
        tracker = ExperimentTracker(sample_config, backend="wandb")
        tracker.run = None

        run_id = tracker.get_run_id()
        assert run_id is None

    def test_get_run_url_wandb(self, sample_config):
        """Test getting run URL with WandB backend."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_run.url = "https://wandb.ai/test/project/runs/test-run"
            mock_wandb.init.return_value = mock_run

            tracker = ExperimentTracker(sample_config, backend="wandb")

            run_url = tracker.get_run_url()
            assert run_url == "https://wandb.ai/test/project/runs/test-run"

    def test_get_run_url_mlflow(self, sample_config):
        """Test getting run URL with MLflow backend."""
        with patch("src.utils.experiment_tracking.mlflow") as mock_mlflow:
            mock_run = Mock()
            mock_mlflow.start_run.return_value = mock_run
            mock_mlflow.get_tracking_uri.return_value = "http://localhost:5000"

            tracker = ExperimentTracker(sample_config, backend="mlflow")

            run_url = tracker.get_run_url()
            assert run_url == "http://localhost:5000"

    def test_get_run_url_no_run(self, sample_config):
        """Test getting run URL when no run is active."""
        tracker = ExperimentTracker(sample_config, backend="wandb")
        tracker.run = None

        run_url = tracker.get_run_url()
        assert run_url is None


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_setup_logging(self, sample_config, temp_dir):
        """Test setting up logging configuration."""
        sample_config.paths.logs_dir = str(temp_dir)

        logger = setup_logging(sample_config)

        assert logger is not None
        assert (temp_dir / "training.log").exists()

    def test_setup_logging_creates_directory(self, sample_config, temp_dir):
        """Test that setup_logging creates logs directory."""
        logs_dir = temp_dir / "new_logs"
        sample_config.paths.logs_dir = str(logs_dir)

        setup_logging(sample_config)

        assert logs_dir.exists()

    def test_save_experiment_config(self, sample_config, temp_dir):
        """Test saving experiment configuration."""
        output_dir = temp_dir / "experiment"

        save_experiment_config(sample_config, output_dir)

        assert output_dir.exists()
        assert (output_dir / "config.yaml").exists()
        assert (output_dir / "config.json").exists()

    def test_save_experiment_config_creates_directory(self, sample_config, temp_dir):
        """Test that save_experiment_config creates output directory."""
        output_dir = temp_dir / "new_experiment" / "subdir"

        save_experiment_config(sample_config, output_dir)

        assert output_dir.exists()

    def test_save_experiment_config_content(self, sample_config, temp_dir):
        """Test that saved configuration files contain correct content."""
        output_dir = temp_dir / "experiment"

        save_experiment_config(sample_config, output_dir)

        # Check JSON file content
        with open(output_dir / "config.json") as f:
            config_data = json.load(f)

        assert "model" in config_data
        assert "training" in config_data
        assert (
            config_data["model"]["pretrained_model_name_or_path"]
            == sample_config.model.pretrained_model_name_or_path
        )


class TestExperimentTrackerEdgeCases:
    """Test edge cases and error handling."""

    def test_log_params_failure(self, sample_config):
        """Test handling parameter logging failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.config.update.side_effect = Exception("Logging error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            # Should not raise an error
            tracker.log_params({"param": "value"})

    def test_log_metrics_failure(self, sample_config):
        """Test handling metrics logging failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.log.side_effect = Exception("Logging error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            # Should not raise an error
            tracker.log_metrics({"loss": 0.5})

    def test_log_artifacts_failure(self, sample_config, temp_dir):
        """Test handling artifacts logging failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.log_artifact.side_effect = Exception("Logging error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            # Should not raise an error
            tracker.log_artifacts(str(temp_dir))

    def test_watch_model_failure(self, sample_config, mock_model):
        """Test handling model watching failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.watch.side_effect = Exception("Watch error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            # Should not raise an error
            tracker.watch_model(mock_model)

    def test_finish_failure(self, sample_config):
        """Test handling finish failure."""
        with patch("src.utils.experiment_tracking.wandb") as mock_wandb:
            mock_run = Mock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.finish.side_effect = Exception("Finish error")

            tracker = ExperimentTracker(sample_config, backend="wandb")

            # Should not raise an error
            tracker.finish()

    def test_flatten_config_with_none_values(self, sample_config):
        """Test flattening config with None values."""
        sample_config.new_key = None

        tracker = ExperimentTracker(sample_config, backend="wandb")
        flattened = tracker._flatten_config(sample_config)

        assert "new_key" in flattened
        assert flattened["new_key"] is None

    def test_flatten_config_with_list_values(self, sample_config):
        """Test flattening config with list values."""
        sample_config.list_key = [1, 2, 3]

        tracker = ExperimentTracker(sample_config, backend="wandb")
        flattened = tracker._flatten_config(sample_config)

        assert "list_key" in flattened
        assert flattened["list_key"] == [1, 2, 3]

    def test_flatten_config_with_dict_values(self, sample_config):
        """Test flattening config with dict values."""
        sample_config.dict_key = {"nested": "value"}

        tracker = ExperimentTracker(sample_config, backend="wandb")
        flattened = tracker._flatten_config(sample_config)

        assert "dict_key" in flattened
        assert flattened["dict_key"] == {"nested": "value"}
