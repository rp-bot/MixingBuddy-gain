"""
Core synthesis orchestration for data generation.

This module coordinates the synthesis pipeline by bringing together
audio processing, gating, error injection, and text generation.
"""

import json
import random
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from tqdm import tqdm
from src.data.audio_processing import load_track_stems, chunk_audio
from src.data.gating import apply_gating
from src.data.error_injection import sample_error_category, apply_gain_error
from src.data.text_generation import create_instruction, create_response
from src.utils.audio_utils import save_audio


def is_silent(audio: np.ndarray, threshold: float) -> bool:
    """Return True if signal amplitude is below threshold.

    Works for mono or multi-channel arrays.
    """
    if audio is None:
        return True
    return float(np.max(np.abs(audio))) < float(threshold)


def sample_target_stem(
    chunk_stems: Dict[str, np.ndarray],
    priors: Dict[str, float],
    rng: random.Random,
) -> str:
    """Sample target stem based on configured probabilities.

    Only considers stems that are actually present in the chunk.
    """
    available_stems = list(chunk_stems.keys())
    available_priors = {stem: priors.get(stem, 0.0) for stem in available_stems}

    # Normalize probabilities
    total = sum(available_priors.values())
    if total == 0:
        # Fallback to uniform if no valid priors
        return rng.choice(available_stems)

    normalized_priors = {k: v / total for k, v in available_priors.items()}

    # Sample using weighted choice
    stems = list(normalized_priors.keys())
    weights = list(normalized_priors.values())
    return rng.choices(stems, weights=weights)[0]


def select_anchor_stem(
    chunk_stems: Dict[str, np.ndarray],
    target_stem: str,
    fallback_order: list,
) -> str:
    """Select anchor stem from fallback order.

    Returns first stem from fallback_order that is:
    - Present in chunk_stems
    - NOT the target_stem
    - NOT "other"
    """
    available_stems = set(chunk_stems.keys())

    for stem in fallback_order:
        if stem in available_stems and stem != target_stem and stem != "other":
            return stem

    # Fallback: select any available stem that isn't target or "other"
    candidates = [s for s in available_stems if s != target_stem and s != "other"]
    if candidates:
        return candidates[0]

    # Last resort: return any stem that isn't target
    candidates = [s for s in available_stems if s != target_stem]
    if candidates:
        return candidates[0]

    # Edge case: only one stem available (should rarely happen)
    return list(available_stems)[0]


def synthesize_chunk(
    chunk_stems: Dict[str, np.ndarray],
    target_stem: str,
    error_category: str,
    config: DictConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Synthesize flawed and reference mixes for a chunk.

    Args:
        chunk_stems: Dict with stem arrays for the chunk
        target_stem: Name of the stem to apply error to
        error_category: Type of error to apply
        config: Configuration object
        rng: Random number generator

    Returns:
        Tuple of (reference_mix, flawed_mix, metadata)
    """
    # Create reference mix by summing all stems
    reference_mix = sum(chunk_stems.values())

    # Apply error to target stem
    modified_stems = chunk_stems.copy()
    if error_category != "no_error":
        modified_stems[target_stem], actual_gain_db = apply_gain_error(
            chunk_stems[target_stem], error_category, config.error, rng
        )
    else:
        actual_gain_db = 0.0

    # Create flawed mix by summing modified stems
    flawed_mix = sum(modified_stems.values())

    # Create metadata
    metadata = {
        "target_stem": target_stem,
        "error_category": error_category,
        "intended_gain_db": actual_gain_db,
        "stems_present": list(chunk_stems.keys()),
    }

    return reference_mix, flawed_mix, metadata


def process_split(
    split_name: str,
    musdb_root: Union[str, Path],
    output_root: Union[str, Path],
    config: DictConfig,
    rng: random.Random,
) -> None:
    """Process all tracks in a split (train or test).

    Args:
        split_name: Name of the split ('train' or 'test')
        musdb_root: Root directory of MUSDB18HQ dataset
        output_root: Root directory for output files
        config: Configuration object
        rng: Random number generator
    """
    musdb_root = Path(musdb_root)
    output_root = Path(output_root)

    # Create output directories
    split_output_dir = output_root / split_name
    flawed_mixes_dir = split_output_dir / "flawed_mixes"
    flawed_mixes_dir.mkdir(parents=True, exist_ok=True)

    # Get list of tracks
    split_dir = musdb_root / split_name
    track_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

    if config.get("limit"):
        track_dirs = track_dirs[: config.limit]

    print(f"Processing {len(track_dirs)} tracks in {split_name} split...")

    # Prepare incremental JSONL writing with resume support
    jsonl_path = split_output_dir / config.output[f"{split_name}_samples"]
    existing_uids = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    uid = rec.get("global_uid")
                    if uid:
                        existing_uids.add(uid)
                except Exception:
                    # Skip malformed lines but continue processing
                    continue

    global_chunk_idx = 0

    use_tqdm = bool(getattr(config.get("logging", {}), "tqdm", False))
    track_iter = (
        tqdm(track_dirs, desc=f"{split_name} tracks", total=len(track_dirs))
        if use_tqdm
        else track_dirs
    )

    for track_idx, track_dir in enumerate(track_iter):
        print(f"Processing track {track_idx + 1}/{len(track_dirs)}: {track_dir.name}")

        # Load track stems
        stems = load_track_stems(track_dir, config.audio.sample_rate)

        # Process chunks
        chunk_iterable = chunk_audio(stems, config.chunk.sec, config.audio.sample_rate)
        if use_tqdm:
            # Estimate total chunks from stem length and chunk size for a standard bar
            any_stem = next(iter(stems.values()))
            num_samples = int(any_stem.shape[-1])
            samples_per_chunk = int(config.chunk.sec * config.audio.sample_rate)
            total_chunks = max(0, num_samples // samples_per_chunk)
            chunk_iterable = tqdm(
                chunk_iterable,
                desc="chunks",
                leave=False,
                total=total_chunks if total_chunks > 0 else None,
            )

        for chunk_idx, chunk_stems in chunk_iterable:
            # Apply gating
            if not apply_gating(chunk_stems, config):
                # print(f"Chunk skipped: failed gating, path: {track_dir.name} chunk_idx: {chunk_idx}")
                continue

            # Enumeration switch (default off if missing)
            enumeration_enabled = bool(
                getattr(config.get("enumeration", {}), "enabled", False)
            )

            # Silence threshold
            silence_threshold = float(getattr(config.audio, "silence_threshold", 1e-5))

            # Common chunk filename
            chunk_filename = f"track{track_idx:03d}_chunk{chunk_idx:03d}"

            if enumeration_enabled:
                # Filter anchors by presence and non-silence
                anchor_candidates = [
                    a
                    for a in getattr(config.anchor, "fallback_order", [])
                    if a in chunk_stems
                    and not is_silent(chunk_stems[a], silence_threshold)
                ]

                error_categories = list(getattr(config.error, "priors", {}).keys())
                if not error_categories:
                    error_categories = [
                        "no_error",
                        "quiet",
                        "very_quiet",
                        "loud",
                        "very_loud",
                    ]

                for anchor_stem in anchor_candidates:
                    # Targets exclude anchor and 'other', must be non-silent
                    target_candidates = [
                        s
                        for s in chunk_stems.keys()
                        if s != anchor_stem
                        and s != "other"
                        and not is_silent(chunk_stems[s], silence_threshold)
                    ]

                    for target_stem in target_candidates:
                        for error_category in error_categories:
                            reference_mix, flawed_mix, metadata = synthesize_chunk(
                                chunk_stems, target_stem, error_category, config, rng
                            )

                            # Paths include anchor/target/error to ensure uniqueness
                            suffix = (
                                f"a-{anchor_stem}_t-{target_stem}_e-{error_category}"
                            )
                            flawed_mix_path = (
                                flawed_mixes_dir
                                / f"{chunk_filename}_{suffix}_flawed.wav"
                            )
                            # Save audio files
                            save_audio(
                                flawed_mix,
                                flawed_mix_path,
                                config.audio.sample_rate,
                                config.audio.bit_depth,
                            )

                            # Instruction and response
                            instruction = create_instruction(
                                config.instruction_templates,
                                config.chunk.sec,
                                list(chunk_stems.keys()),
                                anchor_stem,
                                rng,
                            )
                            response = create_response(
                                config.response_templates,
                                error_category,
                                target_stem,
                                config.error.ranges_db,
                                rng,
                            )

                            sample = {
                                "global_uid": f"{split_name}_{chunk_filename}_{suffix}",
                                "instruction": instruction,
                                "response": response,
                                "flawed_mix_path": str(flawed_mix_path),
                                "meta": {
                                    "track_name": track_dir.name,
                                    "split": split_name,
                                    "chunk_idx": chunk_idx,
                                    "time_ref": {
                                        "start_sec": chunk_idx * config.chunk.sec,
                                        "end_sec": (chunk_idx + 1) * config.chunk.sec,
                                    },
                                    "target_stem": metadata["target_stem"],
                                    "anchor_stem": anchor_stem,
                                    "error_category": metadata["error_category"],
                                    "intended_gain_db": metadata["intended_gain_db"],
                                    "stems_present": metadata["stems_present"],
                                    "paths": {
                                        "stems": {
                                            stem: str(track_dir / f"{stem}.wav")
                                            for stem in chunk_stems.keys()
                                        }
                                    },
                                },
                            }

                            # Incremental write with resume: skip if already exists
                            uid = sample["global_uid"]
                            if uid not in existing_uids:
                                with open(jsonl_path, "a") as f:
                                    f.write(json.dumps(sample) + "\n")
                                existing_uids.add(uid)
                            global_chunk_idx += 1
            else:
                # Original single-sample path
                error_category = sample_error_category(config.error.priors, rng)
                target_stem = sample_target_stem(
                    chunk_stems, config.target_stem.priors, rng
                )
                anchor_stem = select_anchor_stem(
                    chunk_stems, target_stem, config.anchor.fallback_order
                )

                reference_mix, flawed_mix, metadata = synthesize_chunk(
                    chunk_stems, target_stem, error_category, config, rng
                )

                flawed_mix_path = flawed_mixes_dir / f"{chunk_filename}_flawed.wav"

                save_audio(
                    flawed_mix,
                    flawed_mix_path,
                    config.audio.sample_rate,
                    config.audio.bit_depth,
                )

                instruction = create_instruction(
                    config.instruction_templates,
                    config.chunk.sec,
                    list(chunk_stems.keys()),
                    anchor_stem,
                    rng,
                )
                response = create_response(
                    config.response_templates,
                    error_category,
                    target_stem,
                    config.error.ranges_db,
                    rng,
                )

                sample = {
                    "global_uid": f"{split_name}_{chunk_filename}",
                    "instruction": instruction,
                    "response": response,
                    "flawed_mix_path": str(flawed_mix_path),
                    "meta": {
                        "track_name": track_dir.name,
                        "split": split_name,
                        "chunk_idx": chunk_idx,
                        "time_ref": {
                            "start_sec": chunk_idx * config.chunk.sec,
                            "end_sec": (chunk_idx + 1) * config.chunk.sec,
                        },
                        "target_stem": metadata["target_stem"],
                        "anchor_stem": anchor_stem,
                        "error_category": metadata["error_category"],
                        "intended_gain_db": metadata["intended_gain_db"],
                        "stems_present": metadata["stems_present"],
                        "paths": {
                            "stems": {
                                stem: str(track_dir / f"{stem}.wav")
                                for stem in chunk_stems.keys()
                            }
                        },
                    },
                }

                uid = sample["global_uid"]
                if uid not in existing_uids:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(sample) + "\n")
                    existing_uids.add(uid)
                global_chunk_idx += 1

    print(f"Saved/updated samples to {jsonl_path}")
