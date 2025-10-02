#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
from tqdm.auto import tqdm

# Ensure project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunking import chunk_track, iter_track_dirs, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MUSDB18HQ into chunk metadata"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/preprocess_musdb.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("preprocess_musdb")

    musdb_root = Path(cfg.musdb.root)
    out_root = Path(cfg.output.root)
    chunk_sec = float(cfg.chunk.sec)
    frame_ms = float(cfg.frame.ms)
    mixture_gate = float(cfg.gating.mixture_min_rms_dbfs)
    stem_gate = float(cfg.gating.stem_min_rms_dbfs)
    min_active = float(cfg.gating.min_active_frame_ratio)
    fallback_order: List[str] = list(cfg.anchor.fallback_order)

    logger.info(
        "Preprocessing MUSDB | root=%s out=%s chunk=%.1fs frame=%.1fms gates(mix=%.1f, stem=%.1f, active=%.2f)",
        str(musdb_root),
        str(out_root),
        chunk_sec,
        frame_ms,
        mixture_gate,
        stem_gate,
        min_active,
    )

    repo_root = Path(__file__).resolve().parents[1]

    for split in ["train", "test"]:
        logger.info("Processing split: %s", split)
        records = []
        tracks = list(iter_track_dirs(musdb_root, split))
        for track_dir in tqdm(tracks, desc=f"{split} tracks", unit="track"):
            chunk_iter = chunk_track(
                track_dir,
                split,
                chunk_sec=chunk_sec,
                frame_ms=frame_ms,
                gating_mixture_min_rms_dbfs=mixture_gate,
                gating_stem_min_rms_dbfs=stem_gate,
                gating_min_active_ratio=min_active,
                anchor_fallback_order=fallback_order,
                base_path=repo_root,
            )
            for rec in tqdm(
                list(chunk_iter),
                desc=f"{track_dir.name} chunks",
                leave=False,
                unit="chunk",
            ):
                records.append(rec)
        out_path = out_root / split / "metadata.jsonl"
        write_jsonl(records, out_path)
        logger.info("Wrote %d chunk records to %s", len(records), str(out_path))


if __name__ == "__main__":
    main()
