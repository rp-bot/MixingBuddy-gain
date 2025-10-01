#!/usr/bin/env python3
"""
Analyze MUSDB18HQ audio loudness/activity statistics to tune gating thresholds.

Outputs:
- raw.csv: per 10s chunk, per stem metrics (rms_dbfs, active_frame_ratio)
- summary.csv: per split and stem, distribution quantiles
"""

import argparse
import csv
import math
import time
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import pandas as pd
from omegaconf import OmegaConf


STEMS = ["mixture", "vocals", "drums", "bass", "other"]
LOGGER_NAME = "musdb_stats"


@dataclass
class ChunkMetrics:
    split: str
    track: str
    stem: str
    start_sec: float
    end_sec: float
    rms_dbfs: float
    active_frame_ratio: float


def dbfs_from_rms(rms: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(rms, eps))


def compute_rms(x: np.ndarray) -> float:
    if x.ndim == 2:  # (num_samples, channels)
        # Power across channels then average
        return float(np.sqrt(np.mean(np.mean(x.astype(np.float64) ** 2, axis=1))))
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def framewise_rms(x: np.ndarray, frame_len: int) -> np.ndarray:
    n = x.shape[0]
    if x.ndim == 2:
        x = x.mean(axis=1)
    num_frames = max(1, n // frame_len)
    trimmed = x[: num_frames * frame_len]
    frames = trimmed.reshape(num_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return rms


def active_ratio(
    x: np.ndarray, sr: int, frame_ms: float, threshold_dbfs: float
) -> float:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    rms = framewise_rms(x, frame_len)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return float(np.mean(rms_db > threshold_dbfs))


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    # Ensure floating point in [-1, 1]
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data, sr


def iter_track_dirs(musdb_root: Path, splits: Iterable[str]) -> Dict[str, List[Path]]:
    split_to_tracks: Dict[str, List[Path]] = {}
    for split in splits:
        split_dir = musdb_root / split
        if not split_dir.exists():
            split_to_tracks[split] = []
            continue
        tracks = [p for p in split_dir.iterdir() if p.is_dir()]
        split_to_tracks[split] = sorted(tracks)
    return split_to_tracks


def analyze_track(
    track_dir: Path,
    split: str,
    chunk_sec: float,
    frame_ms: float,
    active_thresh_dbfs: float,
) -> List[ChunkMetrics]:
    # Required stems
    files = {
        "mixture": track_dir / "mixture.wav",
        "vocals": track_dir / "vocals.wav",
        "drums": track_dir / "drums.wav",
        "bass": track_dir / "bass.wav",
        "other": track_dir / "other.wav",
    }
    # Load audio
    audio: Dict[str, Tuple[np.ndarray, int]] = {}
    for stem, f in files.items():
        if not f.exists():
            continue
        x, sr = read_audio(f)
        audio[stem] = (x, sr)

    if "mixture" not in audio:
        return []

    # Use mixture length as reference
    mix, sr = audio["mixture"]
    num_samples = mix.shape[0]
    chunk_len = int(sr * chunk_sec)
    metrics: List[ChunkMetrics] = []

    for start in range(0, num_samples, chunk_len):
        end = min(start + chunk_len, num_samples)
        if end - start < max(int(0.5 * chunk_len), 1):
            # drop too-short tail (<50% of chunk)
            break
        start_sec = start / sr
        end_sec = end / sr
        for stem in STEMS:
            if stem not in audio:
                continue
            x, sr_s = audio[stem]
            if sr_s != sr:
                # Skip if mismatched sample rate (shouldn't happen in MUSDB18HQ)
                continue
            x_chunk = x[start:end]
            rms = compute_rms(x_chunk)
            rms_db = dbfs_from_rms(rms)
            ar = active_ratio(x_chunk, sr, frame_ms, active_thresh_dbfs)
            metrics.append(
                ChunkMetrics(
                    split=split,
                    track=track_dir.name,
                    stem=stem,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    rms_dbfs=rms_db,
                    active_frame_ratio=ar,
                )
            )
    return metrics


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    def q(series: pd.Series) -> Dict[str, float]:
        qs = series.quantile(quantiles)
        return {f"p{int(q * 100)}": float(v) for q, v in qs.items()}

    rows = []
    for (split, stem), group in df.groupby(["split", "stem"]):
        entry = {
            "split": split,
            "stem": stem,
        }
        entry.update({f"rms_{k}": v for k, v in q(group["rms_dbfs"]).items()})
        entry.update(
            {f"active_{k}": v for k, v in q(group["active_frame_ratio"]).items()}
        )
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MUSDB18HQ stats for threshold tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/musdb_stats.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Setup logging
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(message)s",
        )

    musdb_root = Path(cfg.musdb.root)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_to_tracks = iter_track_dirs(musdb_root, ["train", "test"])

    rng = random.Random(int(cfg.seed))

    logger.info(
        "Starting analysis | musdb_root=%s output_dir=%s sample_size=%s chunk_sec=%.2f frame_ms=%.1f active_thresh_dbfs=%.1f seed=%s",
        str(musdb_root),
        str(out_dir),
        str(cfg.sample_size),
        float(cfg.chunk.sec),
        float(cfg.frame.ms),
        float(cfg.active_threshold_dbfs),
        str(cfg.seed),
    )

    all_metrics: List[ChunkMetrics] = []
    total_tracks_planned = 0
    total_tracks_done = 0
    total_chunks = 0
    t0 = time.time()
    for split, tracks in split_to_tracks.items():
        if not tracks:
            continue
        sample_n = min(int(cfg.sample_size), len(tracks))
        total_tracks_planned += sample_n
        logger.info("Split=%s | sampling %d/%d tracks", split, sample_n, len(tracks))
        tracks_sample = rng.sample(tracks, sample_n)
        for idx, t in enumerate(tracks_sample, start=1):
            try:
                m_before = len(all_metrics)
                track_metrics = analyze_track(
                    t,
                    split,
                    chunk_sec=float(cfg.chunk.sec),
                    frame_ms=float(cfg.frame.ms),
                    active_thresh_dbfs=float(cfg.active_threshold_dbfs),
                )
                all_metrics.extend(track_metrics)
                added = len(all_metrics) - m_before
                total_chunks += added
                total_tracks_done += 1
                if idx == 1 or idx % 1 == 0:
                    logger.info(
                        "[%s] %d/%d tracks | +%d chunks from %s | totals: tracks=%d/%d, chunks=%d",
                        split,
                        idx,
                        sample_n,
                        added,
                        t.name,
                        total_tracks_done,
                        total_tracks_planned,
                        total_chunks,
                    )
            except Exception as e:
                # Skip problematic tracks; continue
                logger.warning(
                    "Error processing track '%s' in split '%s': %s",
                    t.name,
                    split,
                    str(e),
                )
                continue

    if not all_metrics:
        logger.error("No metrics computed. Check MUSDB root path: %s", str(musdb_root))
        return

    # Write raw CSV
    raw_path = out_dir / "raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "track",
                "stem",
                "start_sec",
                "end_sec",
                "rms_dbfs",
                "active_frame_ratio",
            ]
        )
        for m in all_metrics:
            writer.writerow(
                [
                    m.split,
                    m.track,
                    m.stem,
                    f"{m.start_sec:.3f}",
                    f"{m.end_sec:.3f}",
                    f"{m.rms_dbfs:.3f}",
                    f"{m.active_frame_ratio:.6f}",
                ]
            )

    # Summary CSV
    df = pd.read_csv(raw_path)
    summary_df = summarize(df)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    elapsed = time.time() - t0
    logger.info("Wrote raw metrics to: %s", str(raw_path))
    logger.info("Wrote summary to: %s", str(summary_path))
    logger.info(
        "Completed analysis | tracks=%d/%d chunks=%d elapsed=%.1fs",
        total_tracks_done,
        total_tracks_planned,
        total_chunks,
        elapsed,
    )


if __name__ == "__main__":
    main()
