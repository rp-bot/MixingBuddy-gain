#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf
# from tqdm.auto import tqdm  # Not used in this script

# Ensure project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthesis import load_metadata, load_error_labels, synthesize_training_samples, write_training_samples  # noqa: E402


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
        audio_sr = int(getattr(cfg, "audio", {}).get("sample_rate", 24000))
        limit = getattr(cfg, "limit", None)
        peak_norm = bool(getattr(cfg, "audio", {}).get("peak_normalize", True))
        peak_target = float(getattr(cfg, "audio", {}).get("peak_target", 0.99))
        flawed_mix_dir = output_root / split / flawed_mix_subdir

        samples = list(
            synthesize_training_samples(
                metadata=metadata,
                error_labels=error_labels,
                instruction_templates=instruction_templates,
                response_templates=response_templates,
                seed=seed,
                audio_sample_rate=audio_sr,
                flawed_mix_output_dir=flawed_mix_dir,
                peak_normalize=peak_norm,
                peak_target=peak_target,
                limit=limit,
            )
        )

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
