#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm


def load_metadata(jsonl_path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            r = json.loads(line)
            track_id = r.get("track_id")
            split = r.get("split")
            act = r.get("activity", {})
            stem_rms = act.get("stem_rms_dbfs", {})
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
    return pd.DataFrame(rows)


def aggregate_per_track(df: pd.DataFrame) -> pd.DataFrame:
    # Pool across stems for each track
    grouped = df.groupby(["split", "track_id"])  # pooled stems
    stats = (
        grouped["rms_dbfs"]
        .agg(
            median_track_dbfs="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            count="count",
        )
        .reset_index()
    )
    stats["iqr_db"] = stats["p75"] - stats["p25"]
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-track pooled stem RMS stats (median/IQR)"
    )
    parser.add_argument("--input-root", type=str, default="data/musdb18hq_processed")
    parser.add_argument(
        "--output-root", type=str, default="data/musdb18hq_processed"
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    for split in ["train", "test"]:
        meta_path = input_root / split / "metadata.jsonl"
        if not meta_path.exists():
            continue
        df = load_metadata(meta_path)
        if df.empty:
            continue
        stats = aggregate_per_track(df)
        out_path = output_root / split / "track_stats.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for _, row in tqdm(
                stats.iterrows(), total=len(stats), desc=f"{split} tracks", unit="track"
            ):

                rec = {
                    "split": row["split"],
                    "track_id": row["track_id"],
                    "median_track_dbfs": float(row["median_track_dbfs"]),
                    "iqr_db": float(row["iqr_db"]),
                    "count": int(row["count"]),
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(stats)} track stats to {out_path}")


if __name__ == "__main__":
    main()
