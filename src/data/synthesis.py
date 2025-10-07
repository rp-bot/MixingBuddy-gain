import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import soundfile as sf
from src.utils.audio_utils import db_to_linear, load_audio_chunk, to_mono


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
    *,
    audio_sample_rate: Optional[int] = None,
    flawed_mix_output_dir: Optional[Path] = None,
    peak_normalize: bool = True,
    peak_target: float = 0.99,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """Synthesize training samples from chunks and error labels.

    Optionally generates and saves a flawed mix WAV per sample and includes
    its path in the yielded sample under key 'flawed_mix_path'.
    """
    rng = random.Random(seed)

    count = 0
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

        # Optionally synthesize flawed mix audio
        if audio_sample_rate is not None and flawed_mix_output_dir is not None:
            flawed_mix_output_dir.mkdir(parents=True, exist_ok=True)
            out_path = flawed_mix_output_dir / f"{global_uid}.wav"

            # Load stems chunk
            start_sec = float(chunk_meta["start_sec"])
            end_sec = float(chunk_meta["end_sec"])
            stems_paths: Dict[str, str] = chunk_meta["paths"]["stems"]

            stem_audio: Dict[str, np.ndarray] = {}
            for stem_name, stem_path in stems_paths.items():
                audio = load_audio_chunk(
                    stem_path, start_sec, end_sec, audio_sample_rate
                )
                audio = to_mono(audio)
                stem_audio[stem_name] = audio

            # Align lengths
            max_len = max((a.shape[0] for a in stem_audio.values()), default=0)
            if max_len == 0:
                mix = np.zeros(1, dtype=np.float32)
            else:
                for k, a in list(stem_audio.items()):
                    if a.shape[0] < max_len:
                        pad = np.zeros(max_len - a.shape[0], dtype=np.float32)
                        stem_audio[k] = np.concatenate([a, pad], axis=0)

                # Apply gain to target stem to introduce the flaw
                target_stem = error_label["target_stem"]
                intended_gain_db = float(error_label["intended_gain_db"])
                # We apply the NEGATIVE of intended correction to synthesize the flawed state
                flawed_gain = db_to_linear(-intended_gain_db)

                mix = np.zeros(max_len, dtype=np.float32)
                for stem_name, a in stem_audio.items():
                    g = flawed_gain if stem_name == target_stem else 1.0
                    mix = mix + (a.astype(np.float32) * float(g))

                # Peak normalize to avoid clipping if requested
                if peak_normalize:
                    peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
                    if peak > peak_target:
                        mix = mix / (peak + 1e-12) * peak_target

            # Write WAV
            sf.write(str(out_path), mix, audio_sample_rate, subtype="PCM_16")
            training_sample["flawed_mix_path"] = str(out_path)

        yield training_sample

        count += 1
        if limit is not None and count >= int(limit):
            break


def write_training_samples(
    samples: Iterator[Dict],
    output_path: Path,
) -> None:
    """Write training samples to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
