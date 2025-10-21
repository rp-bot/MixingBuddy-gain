"""
Label extraction metrics for mixing advice generation.

Extracts and evaluates labels from generated text using:
1. Stem name (pattern matching)
2. Magnitude range (regex extraction with normalization)
3. Error detection (zero-shot classifier)
4. Problem severity (zero-shot classifier: quiet/very_quiet/loud/very_loud)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from transformers import pipeline


class LabelExtractionMetric:
    """Extract and evaluate labels from mixing advice text using pretrained classifier."""

    def __init__(
        self,
        *,
        classifier_model: str,
        device: str,
    ):
        """
        Initialize the label extraction metric with zero-shot classifier.

        Args:
            classifier_model: Name of the Hugging Face zero-shot classification model
            device: Device to run model on (cuda/cpu), auto-detected if None
        """
        # Valid stem names
        self.valid_stems = ["vocals", "drums", "bass", "other"]

        # Set device
        self.device = device

        # Initialize zero-shot classifier
        print(f"Loading zero-shot classifier: {classifier_model}")
        self.classifier = pipeline(
            "zero-shot-classification", model=classifier_model, device=self.device
        )
        print(f"Classifier loaded on device: {self.device}")

        # Define candidate labels based on synthesis.yaml templates
        self.error_detection_labels = [
            "the mix is well-balanced and needs no adjustments",
            "the mix needs adjustment",
        ]

        self.problem_severity_labels = [
            "the stem is too quiet",  # quiet
            "the stem is much too quiet and barely audible",  # very_quiet
            "the stem is too loud and overpowering",  # loud
            "the stem is way too loud and dominating the mix",  # very_loud
        ]

        # Mapping from classifier labels to severity categories
        self.severity_label_map = {
            "the stem is too quiet": "quiet",
            "the stem is much too quiet and barely audible": "very_quiet",
            "the stem is too loud and overpowering": "loud",
            "the stem is way too loud and dominating the mix": "very_loud",
        }

    def extract_stem_name(self, text: str) -> Optional[str]:
        """
        Extract stem name from text.

        Args:
            text: Input text to parse

        Returns:
            Extracted stem name or None if not found
        """
        if not text:
            return None

        text_lower = text.lower()

        for stem in self.valid_stems:
            if f"the {stem}" in text_lower or f"{stem} is" in text_lower:
                return stem

        return None

    def extract_magnitude_range(self, text: str) -> Optional[Tuple[float, float]]:
        """
        Extract magnitude range from text and normalize to (min, max) order.

        Args:
            text: Input text to parse

        Returns:
            Tuple of (min_db, max_db) in normalized order, or None if not found
        """
        if not text:
            return None

        # Pattern to match dB ranges like "between 6 and 12 dB" or "3-5 dB"
        patterns = [
            r"between\s+([\d.]+)\s+and\s+([\d.]+)\s+db",
            r"([\d.]+)\s+and\s+([\d.]+)\s+db",
            r"([\d.]+)-([\d.]+)\s+db",
            r"([\d.]+)\s+to\s+([\d.]+)\s+db",
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    val1 = float(match.group(1))
                    val2 = float(match.group(2))
                    # IMPORTANT: Always normalize to (min, max) order
                    return (min(val1, val2), max(val1, val2))
                except ValueError:
                    continue

        return None

    def classify_error_detection(self, text: str) -> bool:
        """
        Classify whether an error is detected in the text.

        Args:
            text: Input text to classify

        Returns:
            True if error detected, False if no error
        """
        if not text:
            return False

        result = self.classifier(
            text, candidate_labels=self.error_detection_labels, multi_label=False
        )

        # Get the highest scoring label
        top_label = result["labels"][0]

        # Return True if "needs adjustment" label wins
        return top_label == "the mix needs adjustment"

    def classify_problem_severity(self, text: str) -> Optional[str]:
        """
        Classify the problem severity using zero-shot classifier.

        Args:
            text: Input text to classify

        Returns:
            One of: "quiet", "very_quiet", "loud", "very_loud", or None
        """
        if not text:
            return None

        result = self.classifier(
            text, candidate_labels=self.problem_severity_labels, multi_label=False
        )

        # Get the highest scoring label
        top_label = result["labels"][0]

        # Map to severity category
        return self.severity_label_map.get(top_label)

    def extract_labels(self, text: str) -> Dict[str, Any]:
        """
        Extract all labels from text.

        Args:
            text: Input text to parse

        Returns:
            Dictionary with extracted labels
        """
        # Extract basic components
        stem_name = self.extract_stem_name(text)
        magnitude_range = self.extract_magnitude_range(text)

        # Classify using zero-shot classifier
        error_detected = self.classify_error_detection(text)

        # Only classify severity if error detected
        problem_severity = None
        if error_detected:
            problem_severity = self.classify_problem_severity(text)

        return {
            "stem_name": stem_name,
            "magnitude_range": magnitude_range,
            "error_detected": error_detected,
            "problem_severity": problem_severity,
        }

    def get_ground_truth_labels(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ground truth labels from prediction metadata.

        Args:
            prediction: Prediction dictionary with metadata

        Returns:
            Dictionary with ground truth labels
        """
        error_category = prediction.get("error_category", "")
        target_stem = prediction.get("target_stem", "")
        min_db = prediction.get("expected_magnitude_min_db")
        max_db = prediction.get("expected_magnitude_max_db")

        # Handle no_error case
        if error_category == "no_error":
            return {
                "stem_name": None,
                "magnitude_range": None,
                "error_detected": False,
                "problem_severity": None,
            }

        # Handle magnitude range (already in correct order from ground truth)
        magnitude_range = None
        if min_db is not None and max_db is not None:
            magnitude_range = (float(min_db), float(max_db))

        return {
            "stem_name": target_stem if target_stem else None,
            "magnitude_range": magnitude_range,
            "error_detected": True,
            "problem_severity": error_category,  # quiet, very_quiet, loud, very_loud
        }

    def compute_binary_metrics(
        self, y_true: List, y_pred: List, labels: List[str]
    ) -> Dict[str, Any]:
        """
        Compute binary classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Label names

        Returns:
            Dictionary with metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0.0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0.0)
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": ["no_error", "error_detected"],
        }

    def compute_multiclass_metrics(
        self, y_true: List, y_pred: List, labels: List[str]
    ) -> Dict[str, Any]:
        """
        Compute multi-class classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            labels: Label names

        Returns:
            Dictionary with metrics
        """
        accuracy = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0.0, labels=labels
        )
        recall_macro = recall_score(
            y_true, y_pred, average="macro", zero_division=0.0, labels=labels
        )
        f1_macro = f1_score(
            y_true, y_pred, average="macro", zero_division=0.0, labels=labels
        )

        # Weighted-averaged metrics
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0.0, labels=labels
        )
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=0.0, labels=labels
        )
        f1_weighted = f1_score(
            y_true, y_pred, average="weighted", zero_division=0.0, labels=labels
        )

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0.0, labels=labels
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0.0, labels=labels
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0.0, labels=labels
        )

        precision_per_class_dict = {
            label: float(p) for label, p in zip(labels, precision_per_class)
        }
        recall_per_class_dict = {
            label: float(r) for label, r in zip(labels, recall_per_class)
        }
        f1_per_class_dict = {
            label: float(f1) for label, f1 in zip(labels, f1_per_class)
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "precision_per_class": precision_per_class_dict,
            "recall_per_class": recall_per_class_dict,
            "f1_per_class": f1_per_class_dict,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": labels,
        }

    def compute(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Compute label extraction metrics on predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Dictionary with metrics results
        """
        print("\n" + "=" * 50)
        print("COMPUTING LABEL EXTRACTION METRICS")
        print("=" * 50)

        # Extract labels from generated text
        extracted_labels = []
        ground_truth_labels = []

        print(f"Processing {len(predictions)} predictions...")
        for i, pred in enumerate(predictions):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(predictions)} predictions")

            # Extract from generated text
            extracted = self.extract_labels(pred["generated"])
            extracted_labels.append(extracted)

            # Get ground truth from metadata
            gt = self.get_ground_truth_labels(pred)
            ground_truth_labels.append(gt)

        print(f"Completed extraction for all {len(predictions)} predictions")

        # Compute stem name accuracy (includes None as a valid class)
        # This penalizes the model for hallucinating stems on no_error samples
        # Convert None to a string label for sklearn compatibility
        stem_gt = [
            gt["stem_name"] if gt["stem_name"] is not None else "none"
            for gt in ground_truth_labels
        ]
        stem_pred = [
            ex["stem_name"] if ex["stem_name"] is not None else "none"
            for ex in extracted_labels
        ]
        stem_accuracy = accuracy_score(stem_gt, stem_pred)

        # Compute magnitude range accuracy (includes None as a valid class)
        # This penalizes the model for hallucinating magnitudes on no_error samples
        mag_gt = [gt["magnitude_range"] for gt in ground_truth_labels]
        mag_pred = [ex["magnitude_range"] for ex in extracted_labels]
        mag_matches = sum(1 for i in range(len(mag_gt)) if mag_gt[i] == mag_pred[i])
        magnitude_accuracy = mag_matches / len(mag_gt) if mag_gt else 0.0

        # Compute error detection metrics (binary classification on all samples)
        error_gt = [gt["error_detected"] for gt in ground_truth_labels]
        error_pred = [ex["error_detected"] for ex in extracted_labels]
        error_detection_metrics = self.compute_binary_metrics(
            error_gt, error_pred, ["no_error", "error"]
        )

        # Compute problem severity metrics (only on samples with errors)
        severity_indices = [
            i for i, gt in enumerate(ground_truth_labels) if gt["error_detected"]
        ]
        if severity_indices:
            severity_gt = [
                ground_truth_labels[i]["problem_severity"] for i in severity_indices
            ]
            severity_pred = [
                extracted_labels[i]["problem_severity"] for i in severity_indices
            ]
            # Convert None to "unknown" for sklearn compatibility
            severity_gt = [s if s is not None else "unknown" for s in severity_gt]
            severity_pred = [s if s is not None else "unknown" for s in severity_pred]

            severity_labels = ["quiet", "very_quiet", "loud", "very_loud", "unknown"]
            severity_metrics = self.compute_multiclass_metrics(
                severity_gt, severity_pred, severity_labels
            )
        else:
            severity_metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        # Derive direction accuracy from severity
        if severity_indices:
            # Map severity to direction
            def severity_to_direction(sev):
                if sev in ["quiet", "very_quiet"]:
                    return "increase"
                elif sev in ["loud", "very_loud"]:
                    return "decrease"
                return "unknown"

            direction_gt = [
                severity_to_direction(ground_truth_labels[i]["problem_severity"])
                for i in severity_indices
            ]
            direction_pred = [
                severity_to_direction(extracted_labels[i]["problem_severity"])
                for i in severity_indices
            ]
            # Convert None to "unknown"
            direction_gt = [d if d is not None else "unknown" for d in direction_gt]
            direction_pred = [d if d is not None else "unknown" for d in direction_pred]

            direction_labels = ["increase", "decrease", "unknown"]
            direction_metrics = self.compute_multiclass_metrics(
                direction_gt, direction_pred, direction_labels
            )
        else:
            direction_metrics = {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        # Count error categories
        error_counts = {}
        for pred in predictions:
            category = pred.get("error_category", "unknown")
            error_counts[category] = error_counts.get(category, 0) + 1

        # Compute overall accuracy (all components correct)
        overall_correct = 0
        for gt, ex in zip(ground_truth_labels, extracted_labels):
            if (
                gt["stem_name"] == ex["stem_name"]
                and gt["magnitude_range"] == ex["magnitude_range"]
                and gt["error_detected"] == ex["error_detected"]
                and gt["problem_severity"] == ex["problem_severity"]
            ):
                overall_correct += 1
        overall_accuracy = overall_correct / len(predictions)

        results = {
            "stem_name_accuracy": stem_accuracy,
            "magnitude_range_accuracy": magnitude_accuracy,
            "error_detection": error_detection_metrics,
            "problem_severity": severity_metrics,
            "direction": direction_metrics,
            "overall_accuracy": overall_accuracy,
            "error_category_counts": error_counts,
            "total_samples": len(predictions),
            "samples_with_errors": len(severity_indices),
            "extracted_labels": extracted_labels,
            "ground_truth_labels": ground_truth_labels,
        }

        # Compute breakdown by error vs no_error samples for insight
        no_error_indices = [
            i for i, gt in enumerate(ground_truth_labels) if not gt["error_detected"]
        ]
        error_indices = [
            i for i, gt in enumerate(ground_truth_labels) if gt["error_detected"]
        ]

        # Stem accuracy breakdown
        if no_error_indices:
            stem_accuracy_no_error = sum(
                1 for i in no_error_indices if stem_gt[i] == stem_pred[i]
            ) / len(no_error_indices)
        else:
            stem_accuracy_no_error = 0.0

        if error_indices:
            stem_accuracy_error = sum(
                1 for i in error_indices if stem_gt[i] == stem_pred[i]
            ) / len(error_indices)
        else:
            stem_accuracy_error = 0.0

        print("\nLabel Extraction Results:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Samples with errors: {results['samples_with_errors']}")
        print(f"  Samples without errors: {len(no_error_indices)}")
        print(f"  Error category distribution: {error_counts}")
        print(f"\n  Stem name accuracy (overall): {stem_accuracy:.4f}")
        print(f"    - On no_error samples: {stem_accuracy_no_error:.4f}")
        print(f"    - On error samples: {stem_accuracy_error:.4f}")
        print(f"  Magnitude range accuracy (overall): {magnitude_accuracy:.4f}")
        print("\n  Error detection:")
        print(f"    Accuracy: {error_detection_metrics['accuracy']:.4f}")
        print(f"    Precision: {error_detection_metrics['precision']:.4f}")
        print(f"    Recall: {error_detection_metrics['recall']:.4f}")
        print(f"    F1: {error_detection_metrics['f1']:.4f}")
        print("\n  Problem severity (on error samples):")
        print(f"    Accuracy: {severity_metrics['accuracy']:.4f}")
        print(
            f"    Precision (macro): {severity_metrics.get('precision_macro', 0.0):.4f}"
        )
        print(f"    Recall (macro): {severity_metrics.get('recall_macro', 0.0):.4f}")
        print(f"    F1 (macro): {severity_metrics.get('f1_macro', 0.0):.4f}")
        print("\n  Direction (derived from severity):")
        print(f"    Accuracy: {direction_metrics['accuracy']:.4f}")
        print(
            f"    Precision (macro): {direction_metrics.get('precision_macro', 0.0):.4f}"
        )
        print(f"    Recall (macro): {direction_metrics.get('recall_macro', 0.0):.4f}")
        print(f"    F1 (macro): {direction_metrics.get('f1_macro', 0.0):.4f}")
        print(f"\n  Overall accuracy (all components): {overall_accuracy:.4f}")

        return results
