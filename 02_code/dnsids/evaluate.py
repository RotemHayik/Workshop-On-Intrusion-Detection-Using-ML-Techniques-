"""Locked-model inference and evaluation. Does not fit or select models."""

import argparse, json, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from joblib import load
from torch.utils.data import DataLoader
from .data import load_tabular, SequenceData, GraphData, graph_collate
from .models import BiLSTM, GraphSAGE
from .train import predict_neural, predictions_frame, seed_everything
from .metrics import metrics, aggregate_recordings, cluster_intervals


def predict_run(root, run_id, split):
    path = Path(root) / "06_results/runs" / run_id
    conf = json.loads((path / "config.json").read_text())
    kind = conf["model"]
    seed_everything(conf["seed"])
    t0 = time.perf_counter()
    if kind in ["rf", "if"]:
        obj = load(path / "model.joblib")
        x, y, m, features = load_tabular(
            root, split, kind == "if", conf.get("drop_features", [])
        )
        if features != obj["features"]:
            raise ValueError("Feature order mismatch")
        x = obj["imputer"].transform(x)
        score = (
            obj["model"].predict_proba(x)[:, 1]
            if kind == "rf"
            else -obj["model"].score_samples(x)
        )
    else:
        dataset = (SequenceData if kind == "bilstm" else GraphData)(root, split)
        loader = DataLoader(
            dataset,
            batch_size=conf["batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=graph_collate if kind == "sage" else None,
        )
        model = (BiLSTM if kind == "bilstm" else GraphSAGE)(**conf["parameters"])
        checkpoint = torch.load(
            path / "model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        score = predict_neural(model, loader)
        y = dataset.y.numpy().astype(int)
        m = dataset.manifest
    frame = predictions_frame(m, y, score)
    return frame, time.perf_counter() - t0


def evaluate_run(root, run_id, splits):
    root = Path(root)
    folder = root / "06_results/final" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads(
        (root / "06_results/runs" / run_id / "validation_metrics.json").read_text()
    )
    rows = []
    source_rows = []
    intervals = {}
    for split in splits:
        target = folder / f"{split}_predictions.csv"
        if target.exists():
            frame = pd.read_csv(target)
            seconds = json.loads((folder / f"{split}_timing.json").read_text())[
                "seconds"
            ]
        else:
            frame, seconds = predict_run(root, run_id, split)
            frame["prediction"] = (
                frame.score >= thresholds["window"]["threshold"]
            ).astype(int)
            frame.to_csv(target, index=False)
            (folder / f"{split}_timing.json").write_text(
                json.dumps({"seconds": seconds, "includes_loading": True}),
                encoding="utf-8",
            )
        for level, data in [
            ("window", frame),
            ("recording", aggregate_recordings(frame)),
        ]:
            t = thresholds[level]["threshold"]
            result = metrics(data.label, data.score, t)
            rows.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "level": level,
                    "inference_seconds_including_load": seconds,
                    **result,
                }
            )
            intervals[split + "_" + level] = cluster_intervals(data, t)
        for dimension in ["source_name", "evaluation_subset", "tool_label"]:
            for value, part in frame.groupby(dimension, dropna=False):
                source_rows.append(
                    {
                        "run_id": run_id,
                        "split": split,
                        "dimension": dimension,
                        "value": str(value),
                        **metrics(
                            part.label, part.score, thresholds["window"]["threshold"]
                        ),
                    }
                )
        print("EVALUATED", run_id, split, "rows", len(frame), flush=True)
        gc.collect()
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(folder / "stratified_metrics.csv", index=False)
    (folder / "confidence_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--selection", required=True)
    p.add_argument("--splits", nargs="+", default=["test", "external_test"])
    a = p.parse_args()
    selected = json.loads(Path(a.selection).read_text())
    for run_id in selected["evaluate_runs"]:
        evaluate_run(a.root, run_id, a.splits)


if __name__ == "__main__":
    main()
