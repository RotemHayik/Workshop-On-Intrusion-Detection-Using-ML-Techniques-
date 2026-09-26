"""Combine measured results and construct source-aware interpretation evidence."""

import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, beta

SOURCES = {
    "Itamar": "Domainator",
    "Nitsan": "GraphTunnel",
    "dns-tunnel-dataset-master - origin - Rotem": "DACA",
    "CTU_Normal_Benign": "CTU Normal",
}


def summarize(root):
    root = Path(root)
    out = root / "06_results"
    selection = json.loads((root / "03_config/selection_lock.json").read_text())
    names = selection["evaluate_runs"]
    metrics = pd.concat(
        [pd.read_csv(out / "final" / n / "metrics.csv") for n in names],
        ignore_index=True,
    )
    metrics.to_csv(out / "performance_matrix.csv", index=False)
    strat = pd.concat(
        [pd.read_csv(out / "final" / n / "stratified_metrics.csv") for n in names],
        ignore_index=True,
    )
    strat["display_value"] = strat.value.replace(SOURCES)
    strat.to_csv(out / "performance_by_source_and_tool.csv", index=False)
    seeds = []
    for kind in ["bilstm", "sage"]:
        base = selection["selected"][kind]
        for name in [base, base + "_seed17", base + "_seed2026"]:
            path = out / "runs" / name / "COMPLETE.json"
            if path.exists():
                r = json.loads(path.read_text())
                seeds.append(
                    {
                        "model": kind,
                        "run_id": name,
                        "seed": r["seed"],
                        "seconds": r["seconds"],
                        **r["validation"]["window"],
                    }
                )
    pd.DataFrame(seeds).to_csv(out / "seed_stability.csv", index=False)
    pd.DataFrame(seeds).groupby("model")[["recall", "fpr", "f1", "auroc", "auprc"]].agg(
        ["mean", "std", "min", "max"]
    ).to_csv(out / "seed_stability_summary.csv")
    # Distribution-shift effect sizes use recording medians to avoid window pseudo-replication.
    train = pd.read_csv(
        root / "04_data/prepared/train/train_tabular_core42_raw.csv"
    ).merge(
        pd.read_csv(root / "04_data/prepared/train/train_sample_manifest.csv")[
            ["sample_id", "split_group_id", "source_name"]
        ],
        on="sample_id",
    )
    features = list(
        pd.read_csv(
            root / "04_data/prepared/train/train_tabular_core42_raw.csv", nrows=0
        ).columns[2:]
    )
    train_group = (
        train.groupby(["label", "split_group_id"])[features].median().reset_index()
    )
    shifts = []
    for split in ["validation", "test", "external_test"]:
        raw = pd.read_csv(
            root / "04_data/prepared" / split / f"{split}_tabular_core42_raw.csv"
        ).merge(
            pd.read_csv(
                root / "04_data/prepared" / split / f"{split}_sample_manifest.csv"
            )[["sample_id", "split_group_id", "source_name"]],
            on="sample_id",
        )
        groups = (
            raw.groupby(["label", "source_name", "split_group_id"])[features]
            .median()
            .reset_index()
        )
        for (label, source), g in groups.groupby(["label", "source_name"]):
            ref = train_group[train_group.label == label]
            for feature in features:
                a = ref[feature].dropna()
                b = g[feature].dropna()
                iqr = a.quantile(0.75) - a.quantile(0.25)
                shifts.append(
                    {
                        "split": split,
                        "label": label,
                        "source": source,
                        "display_source": SOURCES[source],
                        "feature": feature,
                        "train_groups": len(a),
                        "heldout_groups": len(b),
                        "train_median": a.median(),
                        "heldout_median": b.median(),
                        "median_shift_in_train_iqr": (
                            (b.median() - a.median()) / iqr if iqr > 1e-12 else np.nan
                        ),
                        "ks_statistic": ks_2samp(a, b).statistic,
                    }
                )
    pd.DataFrame(shifts).to_csv(
        out / "recording_level_distribution_shift.csv", index=False
    )
    cases = []
    for split in ["validation", "test", "external_test"]:
        errors = pd.read_csv(
            out / "error_analysis" / split / "all_errors_with_features.csv"
        )
        for (run, source, direction), part in errors.groupby(
            ["run_id", "source_name", "error_type"]
        ):
            row = part.sort_values("sample_id").iloc[0]
            cases.append(
                {
                    "split": split,
                    "run_id": run,
                    "source": SOURCES[source],
                    "error": direction,
                    "error_windows": len(part),
                    "error_recordings": part.split_group_id.nunique(),
                    "sample_id": row.sample_id,
                    "tool": row.tool_label,
                    "scenario": row.scenario_type,
                    "score": row.score,
                    "query_length": row.query_name_len_mean,
                    "subdomain_entropy": row.longest_subdomain_entropy_mean,
                    "domain_share": row.top_base_domain_share,
                    "repeated_query_ratio": row.repeated_query_ratio,
                    "graph_nodes": row.graph_nodes,
                }
            )
    pd.DataFrame(cases).to_csv(out / "representative_error_cases.csv", index=False)
    exact = []
    for _, row in metrics[metrics.level == "recording"].iterrows():
        for measure, success, total in [
            ("recall", row.tp, row.tp + row.fn),
            ("fpr", row.fp, row.fp + row.tn),
        ]:
            if total:
                low = (
                    0 if success == 0 else beta.ppf(0.025, success, total - success + 1)
                )
                high = (
                    1
                    if success == total
                    else beta.ppf(0.975, success + 1, total - success)
                )
                exact.append(
                    {
                        "run_id": row.run_id,
                        "split": row.split,
                        "metric": measure,
                        "success": int(success),
                        "denominator": int(total),
                        "lower_95": low,
                        "upper_95": high,
                    }
                )
    pd.DataFrame(exact).to_csv(
        out / "recording_exact_binomial_intervals.csv", index=False
    )
    print(
        "SUMMARIZED", len(metrics), "performance rows;", len(cases), "case categories"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    summarize(a.root)
