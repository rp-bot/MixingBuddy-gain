"""
Tests for evaluation metrics functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.evaluation.metrics import (
    compute_automatic_mixing_metrics,
    compute_text_metrics,
    compute_mixing_specific_metrics,
    extract_mixing_parameters,
    compute_parameter_accuracy,
    compute_technical_term_accuracy,
    compute_parameter_value_accuracy,
)


class TestAutomaticMixingMetrics:
    """Test cases for automatic mixing metrics computation."""

    def test_compute_automatic_mixing_metrics(
        self, sample_predictions, sample_references
    ):
        """Test computing automatic mixing metrics."""
        metrics = compute_automatic_mixing_metrics(
            sample_predictions, sample_references
        )

        # Should include both text and mixing-specific metrics
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Check for text metrics
        assert "bleu" in metrics
        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics

        # Check for mixing-specific metrics
        assert "overall_technical_accuracy" in metrics

    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        metrics = compute_automatic_mixing_metrics([], [])
        assert isinstance(metrics, dict)

    def test_mismatched_lengths(self):
        """Test handling of mismatched prediction/reference lengths."""
        predictions = ["pred1", "pred2"]
        references = ["ref1"]

        # Should not raise an error, but handle gracefully
        metrics = compute_automatic_mixing_metrics(predictions, references)
        assert isinstance(metrics, dict)


class TestTextMetrics:
    """Test cases for text-based metrics."""

    def test_compute_text_metrics_basic(self, sample_predictions, sample_references):
        """Test basic text metrics computation."""
        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert isinstance(metrics, dict)
        assert "bleu" in metrics
        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        assert "bert_score_precision" in metrics
        assert "bert_score_recall" in metrics
        assert "bert_score_f1" in metrics

        # Check that metrics are valid numbers
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)

    @patch("src.evaluation.metrics.sentence_bleu")
    def test_bleu_score_with_nltk(
        self, mock_bleu, sample_predictions, sample_references
    ):
        """Test BLEU score computation with NLTK available."""
        mock_bleu.return_value = 0.5

        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "bleu" in metrics
        assert metrics["bleu"] == 0.5

    @patch("src.evaluation.metrics.sentence_bleu", side_effect=ImportError)
    def test_bleu_score_without_nltk(
        self, mock_bleu, sample_predictions, sample_references
    ):
        """Test BLEU score computation without NLTK."""
        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "bleu" in metrics
        assert metrics["bleu"] == 0.0

    @patch("src.evaluation.metrics.rouge_scorer")
    def test_rouge_score_with_rouge_score(
        self, mock_rouge, sample_predictions, sample_references
    ):
        """Test ROUGE score computation with rouge_score available."""
        # Mock ROUGE scorer
        mock_scorer = Mock()
        mock_score = Mock()
        mock_score.fmeasure = 0.7
        mock_scorer.score.return_value = {
            "rouge1": mock_score,
            "rouge2": mock_score,
            "rougeL": mock_score,
        }
        mock_rouge.RougeScorer.return_value = mock_scorer

        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        assert metrics["rouge1"] == 0.7

    @patch("src.evaluation.metrics.rouge_scorer", side_effect=ImportError)
    def test_rouge_score_without_rouge_score(
        self, mock_rouge, sample_predictions, sample_references
    ):
        """Test ROUGE score computation without rouge_score."""
        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "rouge1" in metrics
        assert "rouge2" in metrics
        assert "rougeL" in metrics
        assert metrics["rouge1"] == 0.0

    @patch("src.evaluation.metrics.score")
    def test_bert_score_with_bert_score(
        self, mock_score, sample_predictions, sample_references
    ):
        """Test BERT score computation with bert_score available."""
        # Mock BERT score
        mock_score.return_value = (
            torch.tensor([0.8, 0.7, 0.9]),
            torch.tensor([0.75, 0.8, 0.85]),
            torch.tensor([0.77, 0.75, 0.87]),
        )

        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "bert_score_precision" in metrics
        assert "bert_score_recall" in metrics
        assert "bert_score_f1" in metrics
        assert isinstance(metrics["bert_score_precision"], float)

    @patch("src.evaluation.metrics.score", side_effect=ImportError)
    def test_bert_score_without_bert_score(
        self, mock_score, sample_predictions, sample_references
    ):
        """Test BERT score computation without bert_score."""
        metrics = compute_text_metrics(sample_predictions, sample_references)

        assert "bert_score_precision" in metrics
        assert "bert_score_recall" in metrics
        assert "bert_score_f1" in metrics
        assert metrics["bert_score_precision"] == 0.0


class TestMixingSpecificMetrics:
    """Test cases for mixing-specific metrics."""

    def test_compute_mixing_specific_metrics(
        self, sample_predictions, sample_references
    ):
        """Test computing mixing-specific metrics."""
        metrics = compute_mixing_specific_metrics(sample_predictions, sample_references)

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Should include parameter accuracy metrics
        assert "overall_technical_accuracy" in metrics

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        metrics = compute_mixing_specific_metrics([], [])
        assert isinstance(metrics, dict)


class TestParameterExtraction:
    """Test cases for parameter extraction."""

    def test_extract_mixing_parameters_eq(self):
        """Test extracting EQ parameters."""
        text = "Apply EQ at 2kHz with 3dB boost and high-pass filter at 80Hz"
        params = extract_mixing_parameters(text)

        assert "eq_frequencies" in params
        assert 2.0 in params["eq_frequencies"]  # 2kHz
        assert 80.0 in params["eq_frequencies"]  # 80Hz

    def test_extract_mixing_parameters_compression(self):
        """Test extracting compression parameters."""
        text = (
            "Use compression with 3:1 ratio, threshold -12dB, attack 3ms, release 100ms"
        )
        params = extract_mixing_parameters(text)

        assert "compression_ratio" in params
        assert "compression_threshold" in params
        assert "compression_attack" in params
        assert "compression_release" in params

        assert params["compression_threshold"] == -12.0
        assert params["compression_attack"] == 3.0
        assert params["compression_release"] == 100.0

    def test_extract_mixing_parameters_reverb(self):
        """Test extracting reverb parameters."""
        text = "Add reverb with 1.2s decay time and room size 0.8"
        params = extract_mixing_parameters(text)

        assert "reverb_decay" in params
        assert "reverb_room_size" in params

        assert params["reverb_decay"] == 1.2
        assert params["reverb_room_size"] == 0.8

    def test_extract_mixing_parameters_volume(self):
        """Test extracting volume parameters."""
        text = "Set volume to -3dB and level at 0.8"
        params = extract_mixing_parameters(text)

        assert "volume_level" in params
        # Should extract the first volume value found
        assert params["volume_level"] in [-3.0, 0.8]

    def test_extract_mixing_parameters_no_params(self):
        """Test extracting parameters from text with no parameters."""
        text = "This is just plain text with no mixing parameters"
        params = extract_mixing_parameters(text)

        assert isinstance(params, dict)
        assert len(params) == 0

    def test_extract_mixing_parameters_case_insensitive(self):
        """Test that parameter extraction is case insensitive."""
        text = "EQ AT 2KHZ, COMPRESSION 3:1 RATIO"
        params = extract_mixing_parameters(text)

        assert "eq_frequencies" in params
        assert "compression_ratio" in params


class TestParameterAccuracy:
    """Test cases for parameter accuracy computation."""

    def test_compute_parameter_accuracy(self):
        """Test computing parameter accuracy."""
        pred_params = [
            {"eq_frequencies": [2.0, 80.0], "compression_ratio": "3:1"},
            {"eq_frequencies": [1.5, 100.0], "compression_ratio": "4:1"},
        ]
        ref_params = [
            {"eq_frequencies": [2.0, 80.0], "compression_ratio": "3:1"},
            {"eq_frequencies": [1.5, 100.0], "compression_ratio": "4:1"},
        ]

        metrics = compute_parameter_accuracy(pred_params, ref_params)

        assert "eq_frequencies_accuracy" in metrics
        assert "compression_ratio_accuracy" in metrics
        assert metrics["eq_frequencies_accuracy"] == 1.0
        assert metrics["compression_ratio_accuracy"] == 1.0

    def test_compute_parameter_accuracy_partial_match(self):
        """Test computing parameter accuracy with partial matches."""
        pred_params = [
            {"eq_frequencies": [2.0, 80.0]},  # Missing compression_ratio
            {"compression_ratio": "4:1"},  # Missing eq_frequencies
        ]
        ref_params = [
            {"eq_frequencies": [2.0, 80.0], "compression_ratio": "3:1"},
            {"eq_frequencies": [1.5, 100.0], "compression_ratio": "4:1"},
        ]

        metrics = compute_parameter_accuracy(pred_params, ref_params)

        assert "eq_frequencies_accuracy" in metrics
        assert "compression_ratio_accuracy" in metrics
        assert metrics["eq_frequencies_accuracy"] == 0.5  # 1 out of 2 correct
        assert metrics["compression_ratio_accuracy"] == 0.5  # 1 out of 2 correct

    def test_compute_parameter_accuracy_no_reference_params(self):
        """Test computing parameter accuracy when reference has no parameters."""
        pred_params = [{"eq_frequencies": [2.0]}]
        ref_params = [{}]  # No parameters in reference

        metrics = compute_parameter_accuracy(pred_params, ref_params)

        assert "eq_frequencies_accuracy" in metrics
        assert metrics["eq_frequencies_accuracy"] == 0.0


class TestTechnicalTermAccuracy:
    """Test cases for technical term accuracy computation."""

    def test_compute_technical_term_accuracy(self):
        """Test computing technical term accuracy."""
        predictions = [
            "Apply EQ and compression to the track",
            "Use reverb and delay effects",
        ]
        references = [
            "Apply equalizer and compressor to the track",
            "Use reverberation and delay effects",
        ]

        metrics = compute_technical_term_accuracy(predictions, references)

        assert "overall_technical_accuracy" in metrics
        assert "eq_accuracy" in metrics
        assert "compression_accuracy" in metrics
        assert "reverb_accuracy" in metrics

        # Should have some accuracy for terms that appear in both
        assert metrics["overall_technical_accuracy"] > 0

    def test_compute_technical_term_accuracy_no_terms(self):
        """Test computing technical term accuracy with no technical terms."""
        predictions = ["This is just plain text"]
        references = ["This is also plain text"]

        metrics = compute_technical_term_accuracy(predictions, references)

        assert "overall_technical_accuracy" in metrics
        assert metrics["overall_technical_accuracy"] == 0.0

    def test_compute_technical_term_accuracy_case_insensitive(self):
        """Test that technical term accuracy is case insensitive."""
        predictions = ["Apply EQ and COMPRESSION"]
        references = ["Apply eq and compression"]

        metrics = compute_technical_term_accuracy(predictions, references)

        assert "eq_accuracy" in metrics
        assert "compression_accuracy" in metrics
        assert metrics["eq_accuracy"] > 0
        assert metrics["compression_accuracy"] > 0


class TestParameterValueAccuracy:
    """Test cases for parameter value accuracy computation."""

    def test_compute_parameter_value_accuracy(self):
        """Test computing parameter value accuracy."""
        pred_params = [
            {"compression_threshold": -12.0, "reverb_decay": 1.2},
            {"compression_threshold": -10.0, "reverb_decay": 1.0},
        ]
        ref_params = [
            {"compression_threshold": -12.0, "reverb_decay": 1.2},
            {"compression_threshold": -10.0, "reverb_decay": 1.0},
        ]

        metrics = compute_parameter_value_accuracy(pred_params, ref_params)

        assert "compression_threshold_mae" in metrics
        assert "compression_threshold_rmse" in metrics
        assert "reverb_decay_mae" in metrics
        assert "reverb_decay_rmse" in metrics

        # Perfect match should have 0 error
        assert metrics["compression_threshold_mae"] == 0.0
        assert metrics["reverb_decay_mae"] == 0.0

    def test_compute_parameter_value_accuracy_with_errors(self):
        """Test computing parameter value accuracy with errors."""
        pred_params = [
            {"compression_threshold": -10.0},  # 2dB off
            {"compression_threshold": -8.0},  # 2dB off
        ]
        ref_params = [
            {"compression_threshold": -12.0},
            {"compression_threshold": -10.0},
        ]

        metrics = compute_parameter_value_accuracy(pred_params, ref_params)

        assert "compression_threshold_mae" in metrics
        assert "compression_threshold_rmse" in metrics

        # Should have some error
        assert metrics["compression_threshold_mae"] > 0

    def test_compute_parameter_value_accuracy_invalid_values(self):
        """Test computing parameter value accuracy with invalid values."""
        pred_params = [
            {"compression_threshold": "invalid"},  # Invalid value
            {"compression_threshold": -10.0},
        ]
        ref_params = [
            {"compression_threshold": -12.0},
            {"compression_threshold": -10.0},
        ]

        metrics = compute_parameter_value_accuracy(pred_params, ref_params)

        assert "compression_threshold_mae" in metrics
        assert "compression_threshold_rmse" in metrics

        # Should handle invalid values gracefully
        assert not np.isnan(metrics["compression_threshold_mae"])

    def test_compute_parameter_value_accuracy_no_matching_params(self):
        """Test computing parameter value accuracy with no matching parameters."""
        pred_params = [{"compression_threshold": -10.0}]
        ref_params = [{"reverb_decay": 1.2}]  # Different parameter

        metrics = compute_parameter_value_accuracy(pred_params, ref_params)

        assert "compression_threshold_mae" in metrics
        assert "reverb_decay_mae" in metrics

        # Should have 0 error when no matching parameters
        assert metrics["compression_threshold_mae"] == 0.0
        assert metrics["reverb_decay_mae"] == 0.0


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""

    def test_metrics_with_none_values(self):
        """Test metrics computation with None values."""
        predictions = [None, "valid prediction"]
        references = ["valid reference", None]

        # Should handle None values gracefully
        metrics = compute_automatic_mixing_metrics(predictions, references)
        assert isinstance(metrics, dict)

    def test_metrics_with_empty_strings(self):
        """Test metrics computation with empty strings."""
        predictions = ["", "valid prediction"]
        references = ["valid reference", ""]

        metrics = compute_automatic_mixing_metrics(predictions, references)
        assert isinstance(metrics, dict)

    def test_metrics_with_very_long_texts(self):
        """Test metrics computation with very long texts."""
        long_text = "word " * 1000  # Very long text
        predictions = [long_text]
        references = [long_text]

        metrics = compute_automatic_mixing_metrics(predictions, references)
        assert isinstance(metrics, dict)
