"""
Evaluation metrics for automatic mixing tasks.
"""

import logging
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


def compute_automatic_mixing_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Compute metrics specific to automatic mixing tasks.

    Args:
        predictions: List of predicted texts
        references: List of reference texts

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic text metrics
    metrics.update(compute_text_metrics(predictions, references))

    # Automatic mixing specific metrics
    metrics.update(compute_mixing_specific_metrics(predictions, references))

    return metrics


def compute_text_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute basic text-based metrics."""
    metrics = {}

    # BLEU score (simplified version)
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoothie = SmoothingFunction().method4

        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)

        metrics["bleu"] = np.mean(bleu_scores)
    except ImportError:
        logger.warning("NLTK not available, skipping BLEU score")
        metrics["bleu"] = 0.0

    # ROUGE score (simplified version)
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)

        for metric, scores in rouge_scores.items():
            metrics[metric] = np.mean(scores)
    except ImportError:
        logger.warning("rouge_score not available, skipping ROUGE scores")
        for metric in ["rouge1", "rouge2", "rougeL"]:
            metrics[metric] = 0.0

    # BERT Score (if available)
    try:
        from bert_score import score

        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        metrics["bert_score_precision"] = P.mean().item()
        metrics["bert_score_recall"] = R.mean().item()
        metrics["bert_score_f1"] = F1.mean().item()
    except ImportError:
        logger.warning("bert_score not available, skipping BERT scores")
        for metric in ["bert_score_precision", "bert_score_recall", "bert_score_f1"]:
            metrics[metric] = 0.0

    return metrics


def compute_mixing_specific_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute metrics specific to automatic mixing tasks."""
    metrics = {}

    # Extract mixing parameters from text
    pred_params = [extract_mixing_parameters(pred) for pred in predictions]
    ref_params = [extract_mixing_parameters(ref) for ref in references]

    # Parameter accuracy
    param_accuracy = compute_parameter_accuracy(pred_params, ref_params)
    metrics.update(param_accuracy)

    # Technical term accuracy
    tech_accuracy = compute_technical_term_accuracy(predictions, references)
    metrics.update(tech_accuracy)

    # Parameter value accuracy
    value_accuracy = compute_parameter_value_accuracy(pred_params, ref_params)
    metrics.update(value_accuracy)

    return metrics


def extract_mixing_parameters(text: str) -> Dict[str, Any]:
    """Extract mixing parameters from text."""
    params = {}

    # EQ parameters
    eq_patterns = [
        r"(\d+(?:\.\d+)?)\s*khz?",
        r"(\d+(?:\.\d+)?)\s*hz",
        r"(\d+(?:\.\d+)?)\s*db",
    ]

    for pattern in eq_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            params["eq_frequencies"] = [float(m) for m in matches]

    # Compression parameters
    comp_patterns = [
        r"ratio\s*(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)\s*ratio",
        r"threshold\s*(\d+(?:\.\d+)?)\s*db",
        r"attack\s*(\d+(?:\.\d+)?)\s*ms",
        r"release\s*(\d+(?:\.\d+)?)\s*ms",
    ]

    for pattern in comp_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            if "ratio" in pattern:
                params["compression_ratio"] = matches[0]
            elif "threshold" in pattern:
                params["compression_threshold"] = float(matches[0])
            elif "attack" in pattern:
                params["compression_attack"] = float(matches[0])
            elif "release" in pattern:
                params["compression_release"] = float(matches[0])

    # Reverb parameters
    reverb_patterns = [
        r"reverb\s*(\d+(?:\.\d+)?)\s*s",
        r"decay\s*(\d+(?:\.\d+)?)\s*s",
        r"room\s*size\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in reverb_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            if "reverb" in pattern or "decay" in pattern:
                params["reverb_decay"] = float(matches[0])
            elif "room" in pattern:
                params["reverb_room_size"] = float(matches[0])

    # Volume/level parameters
    level_patterns = [
        r"(\d+(?:\.\d+)?)\s*db",
        r"volume\s*(\d+(?:\.\d+)?)",
        r"level\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in level_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            params["volume_level"] = float(matches[0])

    return params


def compute_parameter_accuracy(
    pred_params: List[Dict[str, Any]], ref_params: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute accuracy of parameter extraction."""
    metrics = {}

    param_types = [
        "eq_frequencies",
        "compression_ratio",
        "compression_threshold",
        "compression_attack",
        "compression_release",
        "reverb_decay",
        "reverb_room_size",
        "volume_level",
    ]

    for param_type in param_types:
        correct = 0
        total = 0

        for pred, ref in zip(pred_params, ref_params):
            if param_type in ref:  # Only count if reference has this parameter
                total += 1
                if param_type in pred and pred[param_type] == ref[param_type]:
                    correct += 1

        if total > 0:
            metrics[f"{param_type}_accuracy"] = correct / total
        else:
            metrics[f"{param_type}_accuracy"] = 0.0

    return metrics


def compute_technical_term_accuracy(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute accuracy of technical terms."""
    # Common mixing terms
    technical_terms = [
        "eq",
        "equalizer",
        "compression",
        "compressor",
        "reverb",
        "reverberation",
        "delay",
        "chorus",
        "flanger",
        "phaser",
        "distortion",
        "overdrive",
        "high-pass",
        "low-pass",
        "band-pass",
        "notch",
        "shelf",
        "attack",
        "release",
        "threshold",
        "ratio",
        "knee",
        "makeup",
        "wet",
        "dry",
        "mix",
        "send",
        "return",
        "aux",
        "bus",
        "frequency",
        "amplitude",
        "phase",
        "stereo",
        "mono",
        "pan",
    ]

    metrics = {}

    for term in technical_terms:
        pred_mentions = sum(1 for pred in predictions if term.lower() in pred.lower())
        ref_mentions = sum(1 for ref in references if term.lower() in ref.lower())

        if ref_mentions > 0:
            metrics[f"{term}_accuracy"] = (
                min(pred_mentions, ref_mentions) / ref_mentions
            )
        else:
            metrics[f"{term}_accuracy"] = 0.0

    # Overall technical term accuracy
    total_pred_terms = sum(
        sum(1 for term in technical_terms if term.lower() in pred.lower())
        for pred in predictions
    )
    total_ref_terms = sum(
        sum(1 for term in technical_terms if term.lower() in ref.lower())
        for ref in references
    )

    if total_ref_terms > 0:
        metrics["overall_technical_accuracy"] = (
            min(total_pred_terms, total_ref_terms) / total_ref_terms
        )
    else:
        metrics["overall_technical_accuracy"] = 0.0

    return metrics


def compute_parameter_value_accuracy(
    pred_params: List[Dict[str, Any]], ref_params: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute accuracy of parameter values."""
    metrics = {}

    numeric_params = [
        "compression_threshold",
        "compression_attack",
        "compression_release",
        "reverb_decay",
        "reverb_room_size",
        "volume_level",
    ]

    for param_type in numeric_params:
        errors = []

        for pred, ref in zip(pred_params, ref_params):
            if param_type in ref and param_type in pred:
                try:
                    pred_val = float(pred[param_type])
                    ref_val = float(ref[param_type])
                    error = abs(pred_val - ref_val) / max(
                        ref_val, 1e-6
                    )  # Relative error
                    errors.append(error)
                except (ValueError, TypeError):
                    errors.append(1.0)  # Max error for invalid values

        if errors:
            metrics[f"{param_type}_mae"] = np.mean(errors)
            metrics[f"{param_type}_rmse"] = np.sqrt(np.mean([e**2 for e in errors]))
        else:
            metrics[f"{param_type}_mae"] = 0.0
            metrics[f"{param_type}_rmse"] = 0.0

    return metrics
