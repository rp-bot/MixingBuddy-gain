#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import sys

from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Ensure project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.error_labeling import label_errors_for_split  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Label errors for MUSDB chunks using IQR-scaled track policy"
    )
    parser.add_argument(
        "--config", type=str, default="configs/data/04_error_policy.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(cfg.paths.processed_root)
    priors = dict(cfg.error.priors)
    ranges_db = dict(cfg.error.ranges_db)
    gain_limits = dict(cfg.error.gain_limits_db)
    seed = int(cfg.rng.seed)

    for split in ["train", "test"]:
        meta_path = root / split / "metadata.jsonl"
        stats_path = root / split / "track_stats.jsonl"
        out_path = root / split / "error_labels.jsonl"
        if not meta_path.exists() or not stats_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for rec in tqdm(
                label_errors_for_split(
                    meta_path,
                    stats_path,
                    priors=priors,
                    ranges_db=ranges_db,
                    gain_limits_db=gain_limits,
                    seed=seed,
                ),
                desc=f"{split} labels",
                unit="chunk",
            ):
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote labels: {out_path}")


if __name__ == "__main__":
    main()
