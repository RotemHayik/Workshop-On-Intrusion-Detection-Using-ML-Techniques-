"""Trace selected model errors back to exact canonical transaction windows."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .prepare import reference


def extract(root, raw_dir, split):
    root = Path(root)
    out = root / "04_data/raw_examples" / split
    out.mkdir(parents=True, exist_ok=True)
    errors = pd.read_csv(
        root / "06_results/error_analysis" / split / "all_errors_with_features.csv"
    )
    if errors.empty:
        (out / "README.md").write_text(
            "No errors for the evaluated models on this split."
        )
        return
    # Representative case per model, source and error direction; never cherry-pick metrics.
    selected = (
        errors.sort_values("sample_id")
        .groupby(["run_id", "source_name", "error_type"], dropna=False)
        .head(1)
    )
    selected.to_csv(out / "case_index.csv", index=False)
    samples = selected.drop_duplicates("sample_id")
    groups = set(samples.split_group_id)
    chunks = []
    source = Path(raw_dir) / f"{split}_transactions.csv"
    for chunk in pd.read_csv(source, chunksize=50000, low_memory=False):
        part = chunk[chunk.split_group_id.isin(groups)]
        if len(part):
            chunks.append(part)
    data = pd.concat(chunks, ignore_index=True)
    checks = []
    ref = reference()
    for group_id, group in data.groupby("split_group_id", sort=False):
        group = (
            group.assign(
                _timestamp=pd.to_numeric(group.request_timestamp, errors="coerce")
            )
            .sort_values("_timestamp", kind="mergesort")
            .drop(columns="_timestamp")
            .reset_index(drop=True)
        )
        for _, row in samples[samples.split_group_id == group_id].iterrows():
            start = int(row.window_number) * 128
            window = group.iloc[start : start + int(row.window_tx_count)]
            assert (
                ref.stable_sample_id(split, group_id, int(row.window_number))
                == row.sample_id
            )
            assert len(window) == row.window_tx_count
            window.to_csv(out / f"{row.sample_id}.csv", index=False)
            calculated = ref.aggregate_window(window, int(row.window_number))
            features = pd.read_csv(
                root / "04_data/prepared" / split / f"{split}_tabular_raw.csv", nrows=0
            ).columns[2:]
            a = np.array([calculated[f] for f in features], float)
            b = row[features].to_numpy(float)
            same = bool(np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True))
            checks.append(
                {
                    "sample_id": row.sample_id,
                    "transactions": len(window),
                    "all_52_features_match": same,
                    "max_absolute_difference": float(np.nanmax(np.abs(a - b))),
                }
            )
            if not same:
                raise ValueError(
                    "Raw evidence does not reconstruct stored features: "
                    + row.sample_id
                )
    pd.DataFrame(checks).to_csv(out / "reconstruction_checks.csv", index=False)
    print("RAW EVIDENCE", split, "cases", len(samples), "all matched", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--split", required=True)
    a = p.parse_args()
    extract(a.root, a.raw_dir, a.split)
