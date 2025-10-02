import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np


CATEGORIES = ["no_error", "quiet", "very_quiet", "loud", "very_loud"]


@dataclass
class TrackStats:
    median_track_dbfs: float
    iqr_db: float


def load_track_stats(path: Path) -> Dict[str, TrackStats]:
    stats: Dict[str, TrackStats] = {}
    with path.open("r") as f:
        for line in f:
            r = json.loads(line)
            stats[r["track_id"]] = TrackStats(
                median_track_dbfs=float(r["median_track_dbfs"]),
                iqr_db=float(r["iqr_db"]),
            )
    return stats


def sample_category(rng: random.Random, priors: Dict[str, float]) -> str:
    cats = list(priors.keys())
    probs = np.array([priors[c] for c in cats], dtype=np.float64)
    probs = probs / probs.sum()
    choice = rng.choices(cats, weights=probs, k=1)[0]
    return choice


def compute_intended_gain_db(
    category: str,
    chunk_stem_rms_dbfs: float,
    track_median_dbfs: float,
    track_iqr_db: float,
    scales: Dict[str, float],
    gain_limits_db: Dict[str, float],
) -> float:
    if category == "no_error":
        return 0.0
    delta = float(scales.get(category, 0.0)) * float(track_iqr_db)
    if category in ("quiet", "very_quiet"):
        target_dbfs = track_median_dbfs - delta
    elif category in ("loud", "very_loud"):
        target_dbfs = track_median_dbfs + delta
    else:
        target_dbfs = track_median_dbfs
    gain = target_dbfs - float(chunk_stem_rms_dbfs)
    gain = max(
        float(gain_limits_db.get("min", -12.0)),
        min(float(gain_limits_db.get("max", 12.0)), gain),
    )
    return float(gain)


def label_errors_for_split(
    metadata_path: Path,
    track_stats_path: Path,
    priors: Dict[str, float],
    iqr_scales: Dict[str, float],
    gain_limits_db: Dict[str, float],
    seed: int,
) -> Iterator[Dict]:
    rng = random.Random(int(seed))
    stats = load_track_stats(track_stats_path)
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
            # Pull per-chunk current level and per-track baseline
            stem_rms = (r.get("activity", {}).get("stem_rms_dbfs", {}) or {}).get(
                target_stem
            )
            if stem_rms is None:
                continue
            tstats = stats.get(r.get("track_id"))
            if tstats is None:
                continue
            intended_gain_db = compute_intended_gain_db(
                category=category,
                chunk_stem_rms_dbfs=float(stem_rms),
                track_median_dbfs=tstats.median_track_dbfs,
                track_iqr_db=tstats.iqr_db,
                scales=iqr_scales,
                gain_limits_db=gain_limits_db,
            )
            yield {
                "global_uid": r.get("global_uid"),
                "track_id": r.get("track_id"),
                "split": r.get("split"),
                "anchor_stem": anchor,
                "target_stem": target_stem,
                "category": category,
                "intended_gain_db": round(float(intended_gain_db), 3),
                "policy": "iqr_scaled_track_pooled",
            }
