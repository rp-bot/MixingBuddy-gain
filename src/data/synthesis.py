import json
import random
from pathlib import Path
from typing import Dict, Iterator, List


def load_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """Load metadata by global_uid."""
    metadata = {}
    with metadata_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            metadata[record["global_uid"]] = record
    return metadata


def load_error_labels(error_labels_path: Path) -> Dict[str, Dict]:
    """Load error labels by global_uid."""
    labels = {}
    with error_labels_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            labels[record["global_uid"]] = record
    return labels


def synthesize_training_samples(
    metadata: Dict[str, Dict],
    error_labels: Dict[str, Dict],
    instruction_templates: List[str],
    response_templates: Dict[str, List[str]],
    seed: int = 42,
) -> Iterator[Dict]:
    """Synthesize training samples from chunks and error labels."""
    rng = random.Random(seed)

    for global_uid, chunk_meta in metadata.items():
        error_label = error_labels.get(global_uid)
        if error_label is None:
            continue

        # Generate instruction using template
        # TODO: Add more instruction categories beyond basic track description:
        # - Genre-specific instructions (rock, pop, jazz, etc.)
        # - Mixing context (live vs studio, rough mix vs final)
        # - Specific mixing challenges (frequency masking, dynamics, etc.)
        # - Different mixing engineer personas/styles
        instruction_template = rng.choice(instruction_templates)
        instruction = instruction_template.format(
            duration_sec=chunk_meta["duration_sec"],
            stems_present=", ".join(chunk_meta["stems_present"]),
            anchor_stem=chunk_meta["anchor_stem"],
        )

        # Generate response using template
        category = error_label["category"]
        target_stem = error_label["target_stem"]
        intended_gain_db = error_label["intended_gain_db"]

        # Get templates for this category
        category_templates = response_templates.get(
            category, response_templates.get("no_error", [])
        )
        if not category_templates:
            response = "The mix needs adjustment."
        else:
            # Randomly select a template
            template = rng.choice(category_templates)

            # For loud/very_loud, use absolute value since gain is negative
            abs_gain_db = abs(intended_gain_db)

            response = template.format(
                target_stem=target_stem,
                intended_gain_db=intended_gain_db,
                abs_gain_db=abs_gain_db,
            )

        # Create training sample with instruction and response
        training_sample = {
            "global_uid": global_uid,
            "instruction": instruction,
            "response": response,
            "meta": {
                "split": chunk_meta["split"],
                "track_ref": {
                    "album_id": chunk_meta["album_id"],
                    "track_id": chunk_meta["track_id"],
                },
                "time_ref": {
                    "start_sec": chunk_meta["start_sec"],
                    "end_sec": chunk_meta["end_sec"],
                },
                "anchor_stem": chunk_meta["anchor_stem"],
                "target_stem": error_label["target_stem"],
                "error_category": error_label["category"],
                "intended_gain_db": error_label["intended_gain_db"],
                "activity_snapshot": chunk_meta["activity"],
                "paths": chunk_meta["paths"],
            },
        }

        yield training_sample


def write_training_samples(
    samples: Iterator[Dict],
    output_path: Path,
) -> None:
    """Write training samples to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
