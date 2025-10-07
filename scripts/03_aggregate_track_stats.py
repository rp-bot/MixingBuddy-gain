#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm


def load_metadata(jsonl_path: Path) -> (pd.DataFrame, pd.DataFrame):
    rows: List[Dict] = []
    deltas: List[Dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            r = json.loads(line)
            track_id = r.get("track_id")
            split = r.get("split")
            act = r.get("activity", {})
            stem_rms = act.get("stem_rms_dbfs", {}) or {}
            anchor = r.get("anchor_stem")
            # Absolute
            for stem, rms in stem_rms.items():
                if rms is None:
                    continue
                rows.append(
                    {
                        "track_id": track_id,
                        "split": split,
                        "stem": stem,
                        "rms_dbfs": rms,
                    }
                )
            # Anchor-relative deltas (non-anchor minus anchor)
            if anchor is not None and stem_rms.get(anchor) is not None:
                a = float(stem_rms[anchor])
                for stem, rms in stem_rms.items():
                    if stem == anchor or rms is None:
                        continue
                    deltas.append(
                        {
                            "track_id": track_id,
                            "split": split,
                            "delta_db": float(rms) - a,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def aggregate_per_track(df_abs: pd.DataFrame, df_delta: pd.DataFrame) -> pd.DataFrame:
    # Absolute pooled RMS
    if not df_abs.empty:
        g_abs = df_abs.groupby(["split", "track_id"])  # pooled stems
        stats_abs = (
            g_abs["rms_dbfs"]
            .agg(
                median_track_dbfs="median",
                p25=lambda s: s.quantile(0.25),
                p75=lambda s: s.quantile(0.75),
                count_abs="count",
            )
            .reset_index()
        )
        stats_abs["iqr_db"] = stats_abs["p75"] - stats_abs["p25"]
    else:
        stats_abs = pd.DataFrame(
            columns=["split", "track_id", "median_track_dbfs", "iqr_db", "count_abs"]
        )  # empty

    # Anchor-relative deltas pooled per track
    if not df_delta.empty:
        g_delta = df_delta.groupby(["split", "track_id"])  # pooled non-anchor deltas
        stats_delta = (
            g_delta["delta_db"]
            .agg(
                median_delta_db="median",
                dp25=lambda s: s.quantile(0.25),
                dp75=lambda s: s.quantile(0.75),
                count_delta="count",
            )
            .reset_index()
        )
        stats_delta["iqr_delta_db"] = stats_delta["dp75"] - stats_delta["dp25"]
        stats_delta = stats_delta.drop(columns=["dp25", "dp75"])  # compact
    else:
        stats_delta = pd.DataFrame(
            columns=[
                "split",
                "track_id",
                "median_delta_db",
                "iqr_delta_db",
                "count_delta",
            ]
        )  # empty

    # Merge
    stats = pd.merge(stats_abs, stats_delta, on=["split", "track_id"], how="outer")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-track pooled stem RMS stats (median/IQR)"
    )
    parser.add_argument("--input-root", type=str, default="data/musdb18hq_processed")
    parser.add_argument("--output-root", type=str, default="data/musdb18hq_processed")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    for split in ["train", "test"]:
        meta_path = input_root / split / "metadata.jsonl"
        if not meta_path.exists():
            continue
        df_abs, df_delta = load_metadata(meta_path)
        if df_abs.empty and df_delta.empty:
            continue
        stats = aggregate_per_track(df_abs, df_delta)
        out_path = output_root / split / "track_stats.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for _, row in tqdm(
                stats.iterrows(), total=len(stats), desc=f"{split} tracks", unit="track"
            ):
                rec = {
                    "split": row["split"],
                    "track_id": row["track_id"],
                    "median_track_dbfs": float(row["median_track_dbfs"])
                    if "median_track_dbfs" in row and pd.notna(row["median_track_dbfs"])
                    else None,
                    "iqr_db": float(row["iqr_db"])
                    if "iqr_db" in row and pd.notna(row["iqr_db"])
                    else None,
                    "count_abs": int(row["count_abs"])
                    if "count_abs" in row and pd.notna(row["count_abs"])
                    else 0,
                    "median_delta_db": float(row["median_delta_db"])
                    if "median_delta_db" in row and pd.notna(row["median_delta_db"])
                    else None,
                    "iqr_delta_db": float(row["iqr_delta_db"])
                    if "iqr_delta_db" in row and pd.notna(row["iqr_delta_db"])
                    else None,
                    "count_delta": int(row["count_delta"])
                    if "count_delta" in row and pd.notna(row["count_delta"])
                    else 0,
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(stats)} track stats to {out_path}")


if __name__ == "__main__":
    main()
