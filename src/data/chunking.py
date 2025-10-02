import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf


STEMS = ["mixture", "vocals", "drums", "bass", "other"]


@dataclass
class ActivityStats:
    rms_dbfs: float
    active_frame_ratio: float


def dbfs_from_rms(rms: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(rms, eps))


def compute_rms(x: np.ndarray) -> float:
    if x.ndim == 2:
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


def compute_activity(
    x: np.ndarray, sr: int, frame_ms: float, threshold_dbfs: float
) -> ActivityStats:
    rms = compute_rms(x)
    rms_db = dbfs_from_rms(rms)
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    fr = framewise_rms(x, frame_len)
    fr_db = 20.0 * np.log10(np.maximum(fr, 1e-12))
    active_ratio = float(np.mean(fr_db > threshold_dbfs))
    return ActivityStats(rms_dbfs=rms_db, active_frame_ratio=active_ratio)


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.dtype != np.float32 and data.dtype != np.float64:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data, sr


def select_anchor_stem(
    stem_to_stats: Dict[str, ActivityStats],
    min_rms_dbfs: float,
    min_active_ratio: float,
    fallback_order: List[str],
) -> Optional[str]:
    for stem in fallback_order:
        stats = stem_to_stats.get(stem)
        if stats is None:
            continue
        if (
            stats.rms_dbfs >= min_rms_dbfs
            and stats.active_frame_ratio >= min_active_ratio
        ):
            return stem
    return None


def hash_uid(track_id: str, start_sec: float, end_sec: float) -> str:
    raw = f"{track_id}:{start_sec:.3f}:{end_sec:.3f}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def iter_track_dirs(musdb_root: Path, split: str) -> Iterator[Path]:
    split_dir = musdb_root / split
    if not split_dir.exists():
        return iter(())
    for p in sorted(split_dir.iterdir()):
        if p.is_dir():
            yield p


def chunk_track(
    track_dir: Path,
    split: str,
    chunk_sec: float,
    frame_ms: float,
    gating_mixture_min_rms_dbfs: float,
    gating_stem_min_rms_dbfs: float,
    gating_min_active_ratio: float,
    anchor_fallback_order: List[str],
    base_path: Optional[Path] = None,
) -> Iterator[Dict]:
    files = {stem: track_dir / f"{stem}.wav" for stem in STEMS}
    audio: Dict[str, Tuple[np.ndarray, int]] = {}
    for stem, f in files.items():
        if f.exists():
            x, sr = read_audio(f)
            audio[stem] = (x, sr)

    if "mixture" not in audio:
        return

    mix, sr = audio["mixture"]
    num_samples = mix.shape[0]
    chunk_len = int(sr * chunk_sec)

    for start in range(0, num_samples, chunk_len):
        end = min(start + chunk_len, num_samples)
        if end - start < max(int(0.5 * chunk_len), 1):
            break

        # Compute activity for mixture; gate out silent chunks
        x_mix = mix[start:end]
        mix_stats = compute_activity(x_mix, sr, frame_ms, gating_stem_min_rms_dbfs)
        if mix_stats.rms_dbfs < gating_mixture_min_rms_dbfs:
            continue

        stem_stats: Dict[str, ActivityStats] = {}
        for stem, (x, sr_s) in audio.items():
            if sr_s != sr:
                continue
            x_chunk = x[start:end]
            stem_stats[stem] = compute_activity(
                x_chunk, sr, frame_ms, gating_stem_min_rms_dbfs
            )

        anchor = select_anchor_stem(
            stem_stats,
            min_rms_dbfs=gating_stem_min_rms_dbfs,
            min_active_ratio=gating_min_active_ratio,
            fallback_order=anchor_fallback_order,
        )

        stems_present = [s for s in ["vocals", "drums", "bass", "other"] if s in audio]
        uid = hash_uid(track_dir.name, start / sr, end / sr)

        # Build paths relative to base_path if provided (e.g., repo root), else keep as-is
        def to_rel(p: Path) -> str:
            if base_path is not None:
                try:
                    return str(p.relative_to(base_path))
                except Exception:
                    return str(p)
            return str(p)

        record = {
            "global_uid": uid,
            "album_id": track_dir.parent.name,
            "track_id": track_dir.name,
            "split": split,
            "start_sec": round(start / sr, 3),
            "end_sec": round(end / sr, 3),
            "duration_sec": round((end - start) / sr, 3),
            "paths": {
                "mixture": to_rel(files["mixture"]) if "mixture" in audio else None,
                "stems": {
                    s: (to_rel(files[s]) if s in audio else None)
                    for s in ["vocals", "drums", "bass", "other"]
                },
            },
            "stems_present": stems_present,
            "anchor_stem": anchor,
            "activity": {
                "mixture_rms_dbfs": round(mix_stats.rms_dbfs, 3),
                "stem_rms_dbfs": {
                    s: round(stem_stats[s].rms_dbfs, 3) if s in stem_stats else None
                    for s in ["vocals", "drums", "bass", "other"]
                },
                "stem_active_ratio": {
                    s: round(stem_stats[s].active_frame_ratio, 6)
                    if s in stem_stats
                    else None
                    for s in ["vocals", "drums", "bass", "other"]
                },
            },
            "quality_flags": {
                "mixture_pass_gate": mix_stats.rms_dbfs >= gating_mixture_min_rms_dbfs,
                "all_stems_inactive": all(
                    (stem_stats.get(s) is None)
                    or (stem_stats[s].active_frame_ratio < gating_min_active_ratio)
                    for s in ["vocals", "drums", "bass", "other"]
                ),
            },
        }
        yield record


def write_jsonl(records: Iterable[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
