"""
Custom evaluation metrics for mixing advice generation.

Includes:
1. Semantic similarity between generated and ground truth responses
2. Label extraction from generated responses (stem name, direction, magnitude)
3. Label accuracy checking against ground truth metadata
"""

import json
import re
from typing import Dict, List, Optional, Tuple
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class SemanticSimilarityMetric:
    """Compute semantic similarity using sentence transformers."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading semantic similarity model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
        )
        return similarity.item()

    def compute_batch(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute similarities for multiple text pairs.

        Args:
            texts1: List of first texts
            texts2: List of second texts

        Returns:
            List of similarity scores
        """
        if len(texts1) != len(texts2):
            raise ValueError("texts1 and texts2 must have the same length")

        similarities = []
        for t1, t2 in zip(texts1, texts2):
            similarities.append(self.compute(t1, t2))

        return similarities


class LabelExtractor:
    """Extract structured labels from mixing advice text using LLM."""

    def __init__(
        self,
        model_name: str,
        device: str,
    ):
        """
        Args:
            model_name: Name of the LLM to use for extraction
            device: Device to run the model on ('cuda' or 'cpu')
        """
        print(f"Loading label extraction model: {model_name}")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

    def _create_extraction_prompt(self, response_text: str) -> str:
        """Create prompt for label extraction."""
        return f"""Given a mixing advice response, extract the following information and return ONLY a JSON object:
1. stem_name: which instrument/track is mentioned (e.g., "drums", "vocals", "bass", "other")
2. direction: "increase", "decrease", or "balanced"
3. magnitude_min_db: minimum dB value mentioned (as a number, or null if not mentioned)
4. magnitude_max_db: maximum dB value mentioned (as a number, or null if not mentioned)

Response: "{response_text}"

Return ONLY the JSON object, no other text:"""

    def extract(self, response_text: str) -> Dict:
        """
        Extract labels from a response text.

        Args:
            response_text: The generated mixing advice text

        Returns:
            Dictionary with keys: stem_name, direction, magnitude_min_db, magnitude_max_db
        """
        prompt = self._create_extraction_prompt(response_text)

        # Create messages for chat format
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts structured information from text.",
            },
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
            )

        # Decode
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[
            0
        ]

        # Extract JSON from the generated text
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{[^}]+\}", generated_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire response
                extracted = json.loads(generated_text)

            # Ensure all required keys exist
            result = {
                "stem_name": extracted.get("stem_name"),
                "direction": extracted.get("direction"),
                "magnitude_min_db": extracted.get("magnitude_min_db"),
                "magnitude_max_db": extracted.get("magnitude_max_db"),
            }
            return result

        except (json.JSONDecodeError, AttributeError) as e:
            # If extraction fails, return None for all fields
            print(f"Warning: Failed to extract labels from response: {e}")
            print(f"Generated text: {generated_text}")
            return {
                "stem_name": None,
                "direction": None,
                "magnitude_min_db": None,
                "magnitude_max_db": None,
            }


class LabelChecker:
    """Check accuracy of extracted labels against ground truth."""

    def __init__(self):
        """Initialize label checker."""
        pass

    def _normalize_stem_name(self, stem: Optional[str]) -> Optional[str]:
        """Normalize stem names for comparison."""
        if stem is None:
            return None
        stem = stem.lower().strip()
        # Handle common variations
        if stem in ["vocal", "voice"]:
            return "vocals"
        if stem in ["drum", "percussion"]:
            return "drums"
        if stem in ["guitar", "keys", "piano", "synth", "instruments"]:
            return "other"
        return stem

    def _get_ground_truth_direction(self, error_category: str) -> str:
        """Map error category to expected direction."""
        if error_category == "no_error":
            return "balanced"
        elif error_category in ["quiet", "very_quiet"]:
            return "increase"
        elif error_category in ["loud", "very_loud"]:
            return "decrease"
        else:
            return "unknown"

    def _get_ground_truth_magnitude(
        self, error_category: str, intended_gain_db: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get expected magnitude range from error category and intended gain."""
        if error_category == "no_error":
            return None, None

        # The advice should roughly match the magnitude of the error
        # Allow some tolerance based on error category
        if error_category in ["quiet", "loud"]:
            # Expect range around 3-6 dB
            return 3.0, 6.0
        elif error_category in ["very_quiet", "very_loud"]:
            # Expect range around 6-12 dB
            return 6.0, 12.0

        return None, None

    def check_stem_name(self, extracted: Dict, ground_truth_stem: str) -> bool:
        """Check if extracted stem name matches ground truth."""
        extracted_stem = self._normalize_stem_name(extracted.get("stem_name"))
        gt_stem = self._normalize_stem_name(ground_truth_stem)

        if extracted_stem is None or gt_stem is None:
            return False

        return extracted_stem == gt_stem

    def check_direction(
        self, extracted: Dict, ground_truth_error_category: str
    ) -> bool:
        """Check if extracted direction matches ground truth."""
        extracted_direction = extracted.get("direction")
        expected_direction = self._get_ground_truth_direction(
            ground_truth_error_category
        )

        if extracted_direction is None or expected_direction == "unknown":
            return False

        return extracted_direction.lower() == expected_direction.lower()

    def check_magnitude(
        self,
        extracted: Dict,
        ground_truth_error_category: str,
        ground_truth_intended_gain_db: float,
    ) -> bool:
        """Check if extracted magnitude overlaps with expected range."""
        extracted_min = extracted.get("magnitude_min_db")
        extracted_max = extracted.get("magnitude_max_db")

        expected_min, expected_max = self._get_ground_truth_magnitude(
            ground_truth_error_category, ground_truth_intended_gain_db
        )

        # If no error, magnitude should be None or not mentioned
        if ground_truth_error_category == "no_error":
            return extracted_min is None and extracted_max is None

        # If error exists but no magnitude extracted, it's wrong
        if extracted_min is None or extracted_max is None:
            return False

        # Check if there's overlap between extracted and expected ranges
        if expected_min is None or expected_max is None:
            return False

        # Allow some tolerance - check if ranges overlap or are close
        # Ranges overlap if: max1 >= min2 and max2 >= min1
        overlap = extracted_max >= expected_min and expected_max >= extracted_min

        return overlap

    def compute_accuracy(
        self,
        extracted_labels: List[Dict],
        ground_truth_stems: List[str],
        ground_truth_error_categories: List[str],
        ground_truth_intended_gains_db: List[float],
    ) -> Dict[str, float]:
        """
        Compute accuracy for all label types.

        Args:
            extracted_labels: List of extracted label dicts
            ground_truth_stems: List of ground truth stem names
            ground_truth_error_categories: List of ground truth error categories
            ground_truth_intended_gains_db: List of ground truth intended gains

        Returns:
            Dictionary with accuracy scores for each label type and overall
        """
        n = len(extracted_labels)
        if n == 0:
            return {
                "stem_name": 0.0,
                "direction": 0.0,
                "magnitude": 0.0,
                "overall": 0.0,
            }

        stem_correct = 0
        direction_correct = 0
        magnitude_correct = 0

        for i in range(n):
            if self.check_stem_name(extracted_labels[i], ground_truth_stems[i]):
                stem_correct += 1

            if self.check_direction(
                extracted_labels[i], ground_truth_error_categories[i]
            ):
                direction_correct += 1

            if self.check_magnitude(
                extracted_labels[i],
                ground_truth_error_categories[i],
                ground_truth_intended_gains_db[i],
            ):
                magnitude_correct += 1

        return {
            "stem_name": stem_correct / n,
            "direction": direction_correct / n,
            "magnitude": magnitude_correct / n,
            "overall": (stem_correct + direction_correct + magnitude_correct) / (3 * n),
        }
