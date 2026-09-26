"""Portable ingestion adapter and inherited Part 2 feature transformations.

Only this module handles input column names. Training modules consume prepared
arrays with one invariant schema. Raw captures must first be parsed into the
transaction fields specified by the adapter (no implicit PCAP parser).
"""

import argparse, json, sys, importlib
from pathlib import Path
import pandas as pd


def reference():
    folder = Path(__file__).resolve().parents[1] / "part2_reference"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("prepare_chapter6_inputs")


def ingest(source, target, column_map=None):
    """Rename adapter fields and validate canonical labels and recording IDs."""
    ref = reference()
    first = True
    for chunk in pd.read_csv(source, chunksize=50000, low_memory=False):
        chunk = chunk.rename(columns=column_map or {})
        missing = set(ref.TABULAR_INPUT_COLUMNS) - set(chunk.columns)
        if missing:
            raise ValueError("Missing canonical fields: " + str(sorted(missing)))
        if chunk.split_group_id.isna().any():
            raise ValueError("Missing recording boundary")
        labels = chunk.traffic_label.astype(str).str.lower()
        if not labels.isin(["benign", "malicious"]).all():
            raise ValueError("Labels must be benign or malicious")
        chunk["traffic_label"] = labels
        chunk[ref.TABULAR_INPUT_COLUMNS].to_csv(
            target, index=False, mode="w" if first else "a", header=first
        )
        first = False
    if first:
        raise ValueError("Input contains no transactions")


def build(raw_dir, output, root, column_map=None):
    """Rebuild all representations in a NEW output folder; fit only Train."""
    output = Path(output).resolve()
    root = Path(root).resolve()
    raw_dir = Path(raw_dir).resolve()
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory to preserve previous results"
        )
    if output == raw_dir or raw_dir in output.parents:
        raise ValueError("Output must not be inside raw inputs")
    output.mkdir(parents=True)
    ingested = output / "canonical_transactions"
    ingested.mkdir()
    ref = reference()
    ref.OUT = output
    ref.REP = output / "prepared"
    splits = ["train", "validation", "test", "external_test"]
    groups = {}
    for split in splits:
        target = ingested / f"{split}_transactions.csv"
        ingest(raw_dir / f"{split}_transactions.csv", target, column_map)
        pairs = pd.read_csv(
            target, usecols=["split_group_id", "traffic_label"]
        ).drop_duplicates()
        if pairs.split_group_id.duplicated().any():
            raise ValueError("Mixed recording labels")
        groups[split] = set(pairs.split_group_id)
        for previous in splits[: splits.index(split)]:
            if groups[split] & groups[previous]:
                raise ValueError("Recording leakage across splits")
    ts = ref.fit_transaction_scaler(ingested / "train_transactions.csv")
    raw, summary, features = ref.process_split(
        "train", ingested / "train_transactions.csv", ts
    )
    scaler = ref.fit_tabular_scaler(raw, features)
    summaries = [summary]
    core = pd.read_csv(
        root / "04_data/Appendix_D_Core_42_Feature_Schema.csv"
    ).feature.tolist()
    for split in splits:
        if split != "train":
            raw, summary, other = ref.process_split(
                split, ingested / f"{split}_transactions.csv", ts
            )
            if other != features:
                raise ValueError("Feature order differs")
            summaries.append(summary)
        scaled = ref.scale_tabular(raw, features, scaler)
        folder = ref.REP / split
        scaled.to_csv(folder / f"{split}_tabular_scaled.csv", index=False)
        for name, frame in [("raw", raw), ("scaled", scaled)]:
            frame[["sample_id", "label"] + core].to_csv(
                folder / f"{split}_tabular_core42_{name}.csv", index=False
            )
    ref.write_specs(ts, scaler, features)
    (output / "preparation_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--column-map", type=Path)
    a = p.parse_args()
    build(
        a.raw_dir,
        a.output,
        a.root,
        json.loads(a.column_map.read_text()) if a.column_map else None,
    )
