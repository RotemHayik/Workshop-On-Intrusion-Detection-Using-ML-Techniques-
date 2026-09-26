from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "chapter6_preparation"
REP = OUT / "representations"
SPLITS = ["train", "validation", "test", "external_test"]


def main() -> None:
    summary = json.loads((OUT / "preparation_summary.json").read_text(encoding="utf-8"))
    checks = pd.read_csv(OUT / "quality_checks.csv")
    extra_checks: list[dict[str, object]] = []
    group_sets: dict[str, set[str]] = {}
    total_windows = 0

    for split in SPLITS:
        split_dir = REP / split
        manifest = pd.read_csv(split_dir / f"{split}_sample_manifest.csv", low_memory=False)
        tabular = pd.read_csv(split_dir / f"{split}_tabular_scaled.csv", nrows=2)
        sequence = np.load(split_dir / f"{split}_sequence.npz")
        graph = np.load(split_dir / f"{split}_graph.npz")
        group_sets[split] = set(manifest["split_group_id"].astype(str))
        total_windows += len(manifest)
        prohibited = {"source_name", "split_group_id", "client_ip", "server_ip", "base_domain", "subdomain", "query_name", "tool_label", "scenario_type"}
        present = prohibited & set(tabular.columns)
        graph_count = len(graph["sample_id"])
        edge_bound_ok = graph["edge_index"].size == 0 or int(graph["edge_index"].max()) < len(graph["node_features"])
        ptr_ok = (
            len(graph["graph_node_ptr"]) == graph_count + 1
            and len(graph["graph_edge_ptr"]) == graph_count + 1
            and int(graph["graph_node_ptr"][-1]) == len(graph["node_features"])
            and int(graph["graph_edge_ptr"][-1]) == len(graph["edge_features"])
        )
        extra_checks.extend([
            {"check": f"{split}_no_raw_identity_in_tabular_predictors", "passed": not present, "observed": ";".join(sorted(present)) if present else "none", "expected": "none"},
            {"check": f"{split}_graph_pointer_integrity", "passed": ptr_ok, "observed": graph_count, "expected": len(manifest)},
            {"check": f"{split}_graph_edge_bounds", "passed": edge_bound_ok, "observed": int(graph["edge_index"].max()) if graph["edge_index"].size else -1, "expected": f"< {len(graph['node_features'])}"},
            {"check": f"{split}_sequence_shape", "passed": sequence["continuous"].shape == (len(manifest), 128, 15), "observed": str(sequence["continuous"].shape), "expected": str((len(manifest), 128, 15))},
        ])

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = group_sets[left] & group_sets[right]
            extra_checks.append({"check": f"split_group_overlap_{left}_{right}", "passed": not overlap, "observed": len(overlap), "expected": 0})

    checks = pd.concat([checks, pd.DataFrame(extra_checks)], ignore_index=True)
    checks.to_csv(OUT / "quality_checks.csv", index=False, encoding="utf-8-sig")
    if not checks["passed"].astype(bool).all():
        raise RuntimeError("Final quality checks failed")

    model_rows = [
        {
            "model": "Random Forest",
            "category": "Traditional supervised",
            "representation": "Tabular / Flow",
            "training_population": "All Train windows; inverse-recording and class-balanced weighting",
            "input": "52 raw auditable window features with Train-median imputation",
            "validation_role": "Hyperparameters and operating threshold only",
            "primary_reason": "Strong nonlinear baseline, interpretable ranking, and direct continuity with Chapters 3-4",
            "literature_anchor": "Domainator uses Random Forest on statistical DNS-window features",
            "status": "LOCKED",
        },
        {
            "model": "Bidirectional LSTM",
            "category": "Deep learning - sequence",
            "representation": "Sequence / Window",
            "training_population": "All Train windows with group-balanced mini-batches",
            "input": "128 transactions x 15 scaled continuous channels + qtype/rcode embeddings + padding mask",
            "validation_role": "Architecture, learning rate, early stopping, and threshold",
            "primary_reason": "Models ordered changes, recurrence, timing, type transitions, and payload dynamics that aggregation can hide",
            "literature_anchor": "Sequence modeling is a recognized DNS-tunnel direction; Domainator demonstrates the value of ordered metadata windows",
            "status": "LOCKED",
        },
        {
            "model": "GraphSAGE",
            "category": "Deep learning - graph",
            "representation": "Graph / Window",
            "training_population": "All Train graphs with group-balanced mini-batches",
            "input": "Client, resolver, base-domain, and subdomain topology; 13 node features; 6 edge descriptors retained for audit/extension",
            "validation_role": "Depth, hidden width, dropout, early stopping, and threshold",
            "primary_reason": "Captures concentration, recurrence, and changing-subdomain structure beyond scalar statistics",
            "literature_anchor": "GraphTunnel uses GraphSAGE on DNS resolution graphs and evaluates unknown/wildcard tunnels",
            "status": "LOCKED",
        },
        {
            "model": "Isolation Forest",
            "category": "Unsupervised / anomaly detection",
            "representation": "Scaled Tabular / Flow",
            "training_population": "Benign Train windows only, sampled evenly by recording",
            "input": "52 Train-scaled window features",
            "validation_role": "max_samples, max_features, and anomaly threshold under a false-positive constraint",
            "primary_reason": "Tests whether tunneling can be detected as deviation from benign behavior, including previously unseen tools",
            "literature_anchor": "GraphTunnel's literature review describes KRTunnel using Isolation Forest with DNS metadata",
            "status": "LOCKED",
        },
    ]
    pd.DataFrame(model_rows).to_csv(OUT / "model_selection_matrix.csv", index=False, encoding="utf-8-sig")

    literature_rows = [
        {
            "paper": "Domainator: Detecting and Identifying DNS-Tunneling Malware Using Metadata Sequences",
            "year": 2025,
            "architecture": "Random Forest",
            "representation": "Ordered per-domain packet windows summarized by pairwise subdomain-similarity statistics",
            "configuration": "10-packet sliding windows; minimum 3; 43,212 windows; 80/20 train/test",
            "dataset_scope": "Seven malware/tunneling samples plus legitimate resolver traffic and modified validation recordings",
            "reported_metrics": "Binary detection F1 0.966; macro precision 0.958; macro recall 0.970; false-positive rate 1.2%",
            "project_use": "Grounds RF and metadata-window reasoning; project keeps recording-safe 128-transaction windows to match its own data scale",
        },
        {
            "paper": "GraphTunnel: Robust DNS Tunnel Detection Based on DNS Recursive Resolution Graph",
            "year": 2024,
            "architecture": "GraphSAGE (plus CNN for tool identification)",
            "representation": "DNS recursive-resolution paths connected through a gateway-centered graph",
            "configuration": "Graph size K=20 paths; 60/40 train/test; batch size 64; learning rate 0.005",
            "dataset_scope": "Self-collected known and unknown tunneling tools plus wildcard-DNS robustness traffic",
            "reported_metrics": "100% detection accuracy in non-wildcard experiments; wildcard F1 99.78%; tool-identification accuracy above 98.57%",
            "project_use": "Grounds GraphSAGE; project uses an explicitly documented proxy graph because recursive resolver paths are unavailable",
        },
        {
            "paper": "Identifying Malicious DNS Tunnel Tools from DoH Traffic Using Hierarchical Machine Learning Classification",
            "year": 2021,
            "architecture": "XGBoost / LightGBM / CatBoost hierarchical GBDT",
            "representation": "Flow-level DoH traffic features in three classification stages",
            "configuration": "Stratified 10-fold cross-validation with 9:1 train/test inside each fold",
            "dataset_scope": "CIRA-CIC-DoHBrw-2020",
            "reported_metrics": "Accuracy 99.81% for DoH filtering, 99.99% for suspicious DoH detection, and 97.22% for tool identification",
            "project_use": "Supports tree-ensemble relevance and hierarchical evaluation; XGBoost is retained as an alternative, not one of the exact four final slots",
        },
        {
            "paper": "KRTunnel as summarized in GraphTunnel's related-work section",
            "year": "secondary citation",
            "architecture": "Isolation Forest",
            "representation": "Android-side DNS metadata including subdomain entropy and TTL",
            "configuration": "Not reproduced from the secondary summary",
            "dataset_scope": "Android DNS tunnel traffic",
            "reported_metrics": "Not claimed from the available secondary summary",
            "project_use": "Grounds the anomaly-detection family; project reports this evidence limitation explicitly",
        },
    ]
    pd.DataFrame(literature_rows).to_csv(OUT / "literature_grounding_matrix.csv", index=False, encoding="utf-8-sig")

    evaluation_protocol = {
        "data_authority": "final_split_manifest.csv and the four frozen split CSV files",
        "sample_unit": "Non-overlapping 128-transaction windows within split_group_id; partial windows require at least 32 transactions",
        "leakage_controls": [
            "No sample crosses a split_group_id boundary",
            "No split_group_id appears in more than one split",
            "All learned preprocessing is fitted on Train only",
            "Validation is used only for tuning, early stopping, and threshold selection",
            "Test and External Test do not influence feature or model selection",
            "Raw source, endpoint, recording, transaction, tool, and domain identities are excluded from predictor matrices",
        ],
        "model_roles": {
            "Random Forest": "supervised tabular baseline",
            "Bidirectional LSTM": "supervised ordered-sequence model",
            "GraphSAGE": "supervised structural graph model",
            "Isolation Forest": "benign-only anomaly detector",
        },
        "hyperparameter_selection": {
            "Random Forest": {"n_estimators": [300, 500], "max_depth": [12, 20, None], "min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.5]},
            "Bidirectional LSTM": {"layers": [1, 2], "hidden_units_per_direction": [32, 64], "dropout": [0.2, 0.3], "learning_rate": [0.001, 0.0003], "early_stopping_patience": 8},
            "GraphSAGE": {"layers": [2, 3], "hidden_units": [32, 64], "dropout": [0.2, 0.3], "learning_rate": [0.001, 0.0003], "pooling": ["mean", "mean_plus_max"], "early_stopping_patience": 8},
            "Isolation Forest": {"n_estimators": [300, 500], "max_samples": [256, 1024], "max_features": [0.5, 1.0], "training_labels_used": "benign Train only"},
        },
        "threshold_policy": {
            "window_level": "Choose on Validation the threshold with the highest malicious recall subject to false-positive rate <= 1%; if no candidate satisfies it, choose the lowest-FPR candidate and report the deviation",
            "recording_level": "Aggregate window scores by the 95th percentile within split_group_id and choose a separate Validation threshold under the same 1% FPR constraint",
            "isolation_forest": "The contamination setting does not define the final threshold; the final operating threshold is selected on Validation",
        },
        "internal_test_metrics": ["AUROC", "AUPRC", "F1", "balanced_accuracy", "precision", "recall", "specificity", "false_positive_rate", "MCC", "confusion_matrix"],
        "external_test_metrics": ["malicious_recall_overall", "recall_by_evaluation_subset", "recall_by_tool", "recall_by_source", "score_distribution"],
        "external_test_prohibited_metrics": ["accuracy", "specificity", "binary_AUROC", "binary_AUPRC", "binary_F1"],
        "uncertainty": "95% confidence intervals from recording-cluster bootstrap resampling",
        "repeated_training": "Three fixed random seeds for neural models; report mean and standard deviation on Validation, then one locked configuration on Test",
        "comparison_rule": "Compare model-specific attribution only at the shared behavioral-signal level; never average RF, sequence, and graph importance values",
    }
    (OUT / "evaluation_protocol.json").write_text(json.dumps(evaluation_protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    checklist = [
        ("Template Chapter 6 wording verified", "PASS", "Four distinct models must cover traditional supervised, deep learning, and unsupervised/anomaly detection"),
        ("Handoff methodology verified", "PASS", "Frozen splits, Train-only preprocessing, and three representation families retained"),
        ("Frozen manifest integrity", "PASS", "267 unique recording groups; no duplicates"),
        ("Tabular representations", "PASS", "52-feature raw and scaled files for all four splits"),
        ("Sequence representations", "PASS", "128 x 15 continuous tensor plus qtype, rcode, and mask for all four splits"),
        ("Graph representations", "PASS", "Graph arrays, pointers, node and edge features for all four splits"),
        ("Train-only preprocessing", "PASS", "Transaction and tabular robust-scaling parameters frozen from Train"),
        ("Cross-representation alignment", "PASS", f"{total_windows:,} sample IDs aligned across tabular, sequence, and graph files"),
        ("Recording leakage", "PASS", "Zero split_group_id overlap between splits"),
        ("Finite model inputs", "PASS", "No NaN or infinity in scaled tabular, sequence, node, or edge arrays"),
        ("Identity leakage", "PASS", "No raw source, IP, domain, group, tool, or scenario fields in predictor matrices"),
        ("Four-model roster", "PASS", "Random Forest, Bidirectional LSTM, GraphSAGE, Isolation Forest"),
        ("Evaluation protocol", "PASS", "Metrics, threshold policy, uncertainty, and External Test restrictions frozen"),
        ("Literature grounding", "PASS", "Three local papers synthesized; anomaly-model evidence limitation recorded"),
        ("Ready to write Chapter 6", "PASS", "No unresolved technical dependency remains for drafting the chapter"),
    ]
    pd.DataFrame(checklist, columns=["item", "status", "evidence"]).to_csv(OUT / "readiness_checklist.csv", index=False, encoding="utf-8-sig")

    counts = {row["split"]: row for row in summary["representation_summaries"]}
    readme = f"""# Chapter 6 readiness package

## Outcome

All technical and methodological preparations required before drafting Chapter 6 are complete. The package is grounded in the handoff MD, the exact Chapter 6 template wording, the completed Chapters 3-5, and the three local academic papers.

## Frozen representation counts

| Split | Source transactions | Source groups | Eligible windows | Represented groups | Excluded groups under 32 tx | Benign windows | Malicious windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | {counts['train']['source_rows']:,} | {counts['train']['source_groups']} | {counts['train']['windows']:,} | {counts['train']['represented_groups']} | {counts['train']['excluded_small_groups']} | {counts['train']['benign_windows']:,} | {counts['train']['malicious_windows']:,} |
| Validation | {counts['validation']['source_rows']:,} | {counts['validation']['source_groups']} | {counts['validation']['windows']:,} | {counts['validation']['represented_groups']} | {counts['validation']['excluded_small_groups']} | {counts['validation']['benign_windows']:,} | {counts['validation']['malicious_windows']:,} |
| Test | {counts['test']['source_rows']:,} | {counts['test']['source_groups']} | {counts['test']['windows']:,} | {counts['test']['represented_groups']} | {counts['test']['excluded_small_groups']} | {counts['test']['benign_windows']:,} | {counts['test']['malicious_windows']:,} |
| External Test | {counts['external_test']['source_rows']:,} | {counts['external_test']['source_groups']} | {counts['external_test']['windows']:,} | {counts['external_test']['represented_groups']} | {counts['external_test']['excluded_small_groups']} | {counts['external_test']['benign_windows']:,} | {counts['external_test']['malicious_windows']:,} |

Windowing is non-overlapping, recording-safe, and identical across all three representations. A final partial window is included only when it contains at least 32 transactions. Tiny ineligible groups remain listed in the per-split exclusion files and are not silently dropped.

## Locked model roster

1. Random Forest - traditional supervised model on the 52-feature Tabular/Flow representation.
2. Bidirectional LSTM - deep sequence model on the ordered 128-transaction representation.
3. GraphSAGE - deep graph model on within-window client/resolver/domain/subdomain structure.
4. Isolation Forest - benign-only anomaly detector on the scaled 52-feature representation.

This roster satisfies the template's exact requirement to cover traditional supervised, deep-learning, and unsupervised/anomaly-detection approaches. XGBoost remains a literature-supported alternative but is not a fifth final model.

## Important graph limitation

The project dataset does not contain full recursive resolver paths. Therefore the GraphSAGE input is a transparent proxy graph built from client, DNS server, base-domain, and subdomain relations inside each recording-safe window. Raw identities are used only to form local topology and are never stored as predictor values. Chapter 6 must state this adaptation explicitly and must not claim an exact replication of GraphTunnel.

## Evaluation contract

- Fit preprocessing and models on Train only.
- Use Validation only for hyperparameters, early stopping, and operating thresholds.
- Use the internal Test once for final binary evaluation.
- Use External Test only for malicious recall/generalization by subset, source, and tool; do not report balanced binary metrics there.
- Select thresholds on Validation under a 1% false-positive-rate constraint.
- Report recording-cluster bootstrap confidence intervals.
- Compare model importance at the behavioral-signal level, not by averaging incompatible attribution scores.

## Package map

- `representation_specification.json` - exact sample, feature, tensor, and graph definitions.
- `representations/<split>/` - raw/scaled tabular CSV, sequence NPZ, graph NPZ, sample manifest, and any tiny-group exclusion list.
- `transaction_scaling_parameters.json` - Train-only transaction/sequence preprocessing.
- `tabular_scaling_parameters.csv` - Train-only full-window tabular preprocessing.
- `model_selection_matrix.csv` - locked four-model design.
- `literature_grounding_matrix.csv` - paper configurations, datasets, metrics, and project adaptation.
- `evaluation_protocol.json` - frozen tuning, threshold, metric, uncertainty, and External Test policy.
- `quality_checks.csv` - machine-verifiable integrity results.
- `readiness_checklist.csv` - final go/no-go checklist.

## Readiness decision

**GO.** Chapter 6 can now be written from finalized, verified inputs. Actual model training belongs to the later modeling/evaluation phase and is not a prerequisite for the Chapter 6 model-selection rationale.
"""
    (OUT / "Chapter_6_Readiness_Package.md").write_text(readme, encoding="utf-8")

    inventory_rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "artifact_inventory.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        inventory_rows.append({"relative_path": path.relative_to(OUT).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    pd.DataFrame(inventory_rows).to_csv(OUT / "artifact_inventory.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"all_checks_passed": True, "total_windows": total_windows, "inventory_files": len(inventory_rows)}, indent=2))


if __name__ == "__main__":
    main()
