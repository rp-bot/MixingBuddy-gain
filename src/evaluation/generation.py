import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm


logger = logging.getLogger(__name__)


def generate_and_compare(
    model: Any,
    dataset: Any,
    num_samples: int,
    max_new_tokens: int,
    use_instruction: bool,
    system_message: str,
    output_dir: Path,
):
    """Generate responses for samples and save to JSONL for later analysis."""
    logger.info("Generating samples for qualitative evaluation")

    predictions = []

    def get_expected_magnitude_range(error_category):
        if error_category == "no_error":
            return None, None
        if error_category in ["quiet", "loud"]:
            return 3.0, 6.0
        if error_category in ["very_quiet", "very_loud"]:
            return 6.0, 12.0
        return None, None

    total_samples = num_samples if num_samples is not None else len(dataset)

    for i in tqdm(range(total_samples), desc="Generating predictions"):
        sample = dataset[i]
        instruction = sample["instruction"]
        audio = sample["audio"]
        ground_truth = sample["response"]
        global_uid = sample["global_uid"]
        target_stem = sample["target_stem"]
        error_category = sample["error_category"]

        item_data = dataset.data[i]
        intended_gain_db = item_data["meta"].get("intended_gain_db", 0.0)
        start_sec = item_data["meta"].get("time_ref", {}).get("start_sec", 0.0)
        end_sec = item_data["meta"].get("time_ref", {}).get("end_sec", 0.0)

        expected_min_db, expected_max_db = get_expected_magnitude_range(error_category)

        text_for_generation = instruction if use_instruction else ""

        generated_text = model.generate(
            text_input=text_for_generation,
            audio=audio,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
        )

        prediction = {
            "global_uid": global_uid,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "generated": generated_text,
            "target_stem": target_stem,
            "error_category": error_category,
            "intended_gain_db": intended_gain_db,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "expected_magnitude_min_db": expected_min_db,
            "expected_magnitude_max_db": expected_max_db,
        }
        predictions.append(prediction)

        if i < 3:
            logger.debug(
                "Sample %d/%d: uid=%s stem=%s err=%s",
                i + 1,
                total_samples,
                global_uid,
                target_stem,
                error_category,
            )

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / "predictions.jsonl"

    logger.info("Saving predictions to %s", predictions_file)
    with open(predictions_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info("Saved %d predictions", len(predictions))
    return predictions_file
