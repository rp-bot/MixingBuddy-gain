#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Ensure project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthesis import (
    load_metadata,
    load_error_labels,
    synthesize_training_samples,
    write_training_samples,
)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize training samples from chunks and error labels"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/05_synthesis.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    processed_root = Path(cfg.paths.processed_root)
    output_root = Path(cfg.paths.output_root)
    seed = int(cfg.rng.seed)

    splits = list(getattr(cfg, "splits", ["train", "test"]))
    flawed_mix_subdir = getattr(cfg, "output", {}).get(
        "flawed_mix_subdir", "flawed_mixes"
    )

    for split in splits:
        print(f"Processing {split} split...")

        metadata_path = processed_root / split / "metadata.jsonl"
        error_labels_path = processed_root / split / "error_labels.jsonl"

        if not metadata_path.exists() or not error_labels_path.exists():
            print(f"Skipping {split} - missing metadata or error labels")
            continue

        # Load data
        metadata = load_metadata(metadata_path)
        error_labels = load_error_labels(error_labels_path)

        print(f"Loaded {len(metadata)} chunks, {len(error_labels)} error labels")

        # Synthesize training samples (with flawed mix generation)
        instruction_templates = list(cfg.instruction_templates)
        response_templates = dict(cfg.response_templates)
        audio_sr = int(
            getattr(cfg, "audio", {}).get("sample_rate", 48000)
        )  # Default to 48kHz
        audio_bit_depth = int(
            getattr(cfg, "audio", {}).get("bit_depth", 32)
        )  # Default to 32-bit
        limit = getattr(cfg, "limit", None)
        peak_norm = bool(
            getattr(cfg, "audio", {}).get("peak_normalize", False)
        )  # Default to False
        peak_target = float(getattr(cfg, "audio", {}).get("peak_target", 0.99))
        flawed_mix_dir = output_root / split / flawed_mix_subdir

        # Create progress bar for synthesis
        total_samples = min(len(metadata), len(error_labels))
        if limit is not None:
            total_samples = min(total_samples, limit)

        samples = []
        with tqdm(total=total_samples, desc=f"Synthesizing {split} samples") as pbar:
            for sample in synthesize_training_samples(
                metadata=metadata,
                error_labels=error_labels,
                instruction_templates=instruction_templates,
                response_templates=response_templates,
                seed=seed,
                audio_sample_rate=audio_sr,
                audio_bit_depth=audio_bit_depth,
                flawed_mix_output_dir=flawed_mix_dir,
                peak_normalize=peak_norm,
                peak_target=peak_target,
                limit=limit,
            ):
                samples.append(sample)
                pbar.update(1)

        # Write output
        out_cfg = getattr(cfg, "output", {})
        if split == "train":
            output_name = out_cfg.get("train_samples", "training_samples.jsonl")
        else:
            output_name = out_cfg.get("test_samples", "test_samples.jsonl")

        output_path = output_root / split / output_name
        write_training_samples(samples, output_path)

        print(f"Wrote {len(samples)} {split}ing samples to {output_path}")


if __name__ == "__main__":
    main()
