"""Sample-level error tables, feature evidence, and model disagreements."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def diagnose(root, run_ids, split="validation", final=False):
    root = Path(root)
    out = root / "06_results/error_analysis" / split
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(root / "04_data/prepared" / split / f"{split}_tabular_raw.csv")
    tables = []
    frames = {}
    for run_id in run_ids:
        p = (
            (root / "06_results/final" / run_id / f"{split}_predictions.csv")
            if final
            else (root / "06_results/runs" / run_id / "validation_predictions.csv")
        )
        f = pd.read_csv(p)
        frames[run_id] = f
        f["error_type"] = np.select(
            [
                (f.label == 0) & (f.prediction == 1),
                (f.label == 1) & (f.prediction == 0),
            ],
            ["FP", "FN"],
            default="correct",
        )
        wrong = f[f.error_type != "correct"].merge(
            raw.drop(columns="label"), on="sample_id", validate="one_to_one"
        )
        wrong.insert(0, "run_id", run_id)
        tables.append(wrong)
        wrong.to_csv(out / f"{run_id}_errors.csv", index=False)
        summaries = []
        for dimension in [
            "source_name",
            "tool_label",
            "scenario_type",
            "evaluation_subset",
        ]:
            for value, g in f.groupby(dimension, dropna=False):
                counts = g.error_type.value_counts()
                pos = (g.label == 1).sum()
                neg = (g.label == 0).sum()
                summaries.append(
                    {
                        "run_id": run_id,
                        "dimension": dimension,
                        "value": str(value),
                        "n": len(g),
                        "fp": counts.get("FP", 0),
                        "fn": counts.get("FN", 0),
                        "fpr": counts.get("FP", 0) / neg if neg else np.nan,
                        "fnr": counts.get("FN", 0) / pos if pos else np.nan,
                    }
                )
        pd.DataFrame(summaries).to_csv(out / f"{run_id}_categories.csv", index=False)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            out / "all_errors_with_features.csv", index=False
        )
    joined = None
    for name, f in frames.items():
        block = f[["sample_id", "label", "score", "prediction"]].rename(
            columns={"score": name + "_score", "prediction": name + "_prediction"}
        )
        joined = (
            block
            if joined is None
            else joined.merge(
                block.drop(columns="label"), on="sample_id", validate="one_to_one"
            )
        )
    cols = [c for c in joined if c.endswith("_prediction")]
    joined["disagreement"] = joined[cols].nunique(axis=1) > 1
    joined.to_csv(out / "aligned_model_predictions.csv", index=False)
    joined[joined.disagreement].merge(raw.drop(columns="label"), on="sample_id").to_csv(
        out / "disagreements_with_features.csv", index=False
    )
    print(
        "DIAGNOSED", split, "disagreements", int(joined.disagreement.sum()), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    diagnose(a.root, a.runs, a.split, a.final)
