import json
import random
from pathlib import Path
from typing import Dict, Iterator

import numpy as np


CATEGORIES = ["no_error", "quiet", "very_quiet", "loud", "very_loud"]


# Track statistics are no longer needed with the fixed range approach


def sample_category(rng: random.Random, priors: Dict[str, float]) -> str:
    cats = list(priors.keys())
    probs = np.array([priors[c] for c in cats], dtype=np.float64)
    probs = probs / probs.sum()
    choice = rng.choices(cats, weights=probs, k=1)[0]
    return choice


def compute_intended_gain_db(
    category: str,
    ranges_db: Dict[str, list],
    gain_limits_db: Dict[str, float],
    rng: random.Random,
) -> float:
    """Compute intended gain using fixed ranges for each category."""
    if category == "no_error":
        return 0.0

    # Get the range for this category
    range_db = ranges_db.get(category, [0, 0])
    if len(range_db) != 2:
        return 0.0

    # Randomly select a value within the range
    min_db, max_db = range_db
    gain_db = rng.uniform(min_db, max_db)

    # Apply overall gain limits for safety
    gain_db = max(
        float(gain_limits_db.get("min", -15.0)),
        min(float(gain_limits_db.get("max", 15.0)), gain_db),
    )
    return float(gain_db)


def label_errors_for_split(
    metadata_path: Path,
    track_stats_path: Path,  # Kept for compatibility but not used
    priors: Dict[str, float],
    ranges_db: Dict[str, list],
    gain_limits_db: Dict[str, float],
    seed: int,
) -> Iterator[Dict]:
    """Label errors using straightforward fixed ranges for each category."""
    rng = random.Random(int(seed))

    with metadata_path.open("r") as f:
        for line in f:
            r = json.loads(line)
            anchor = r.get("anchor_stem")
            if anchor is None:
                continue
            candidates = [
                s for s in ["vocals", "drums", "bass", "other"] if s != anchor
            ]
            if not candidates:
                continue
            target_stem = rng.choice(candidates)
            category = sample_category(rng, priors)

            # Compute intended gain using fixed ranges
            intended_gain_db = compute_intended_gain_db(
                category=category,
                ranges_db=ranges_db,
                gain_limits_db=gain_limits_db,
                rng=rng,
            )

            yield {
                "global_uid": r.get("global_uid"),
                "track_id": r.get("track_id"),
                "split": r.get("split"),
                "anchor_stem": anchor,
                "target_stem": target_stem,
                "category": category,
                "intended_gain_db": round(float(intended_gain_db), 3),
                "policy": "fixed_ranges",
            }
