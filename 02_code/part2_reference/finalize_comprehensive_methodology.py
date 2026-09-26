from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "chapter6_preparation"
OUTPUT_DIR = ROOT / "comprehensive_corrections" / "chapter6_methodology"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((SOURCE_DIR / "evaluation_protocol.json").read_text(encoding="utf-8"))
    protocol["sample_population_distinction"] = {
        "diagnostic_chapters_4_5": "2,362 capped Train windows (maximum 24 per recording) used only for balanced diagnostics, ranking, and distribution-shift visualization",
        "production_models": "23,207 uncapped eligible windows across frozen splits, including 10,971 Train windows; recording/class weighting or group-balanced batches control dominance",
        "scaler_fit": "Final production preprocessing parameters are fitted on all 10,971 eligible Train windows after the core schema is locked",
    }
    protocol["feature_roster"] = {
        "audited_columns": 52,
        "primary_tabular_core": 42,
        "supporting_or_ablation_only": 10,
        "selection_rule": "A field enters the primary core when group-median evidence has Holm-adjusted p<0.05 and |Cliff's delta|>=0.147 in pooled or within-Nitsan analysis; supporting fields remain available only for pre-registered ablation or model-specific structural use",
    }
    protocol["model_semantics"] = {
        "Bidirectional LSTM": "Completed-window classifier: both temporal directions are available only after a 128-transaction window (or an eligible masked partial window) is complete; it is not claimed as per-query streaming detection",
        "GraphSAGE": "The primary baseline uses node attributes and topology only. Stored edge attributes are excluded from vanilla GraphSAGE and reserved for a separately labeled edge-aware ablation",
        "Isolation Forest": "Uses the scaled 42-feature core and is fitted only on benign Train groups sampled or weighted evenly by recording",
    }
    protocol["threshold_policy"]["recording_level_discreteness"] = (
        "Validation contains 11 benign represented recordings. Therefore a 1% empirical recording-level FPR is unattainable except at zero false positives; report the exact observed fraction and a recording-cluster/bootstrap or binomial interval rather than claiming precise 1% calibration."
    )
    protocol["quality_controls_added"] = [
        "Chapter 3 statistical inference uses split_group_id recording medians as independent units",
        "Holm family-wise correction is applied separately across all 52 pooled and within-Nitsan tests",
        "Chapter 4 median imputation and class/group weights are fitted independently inside each training fold",
        "Chapter 4 permutation importance is averaged across all five held-recording folds with five repeats per fold",
    ]
    (OUTPUT_DIR / "evaluation_protocol_revised.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    literature_rows = [
        {
            "paper": "Domainator: Detecting and Identifying DNS-Tunneling Malware Using Metadata Sequences",
            "year": 2025,
            "architecture": "Random Forest",
            "configuration": "10-packet windows (minimum 3); 43,212 windows; 80/20 split",
            "dataset_scope": "Seven tunneling/malware samples plus legitimate resolver traffic",
            "reported_metrics": "Binary F1 0.966; macro precision 0.958; macro recall 0.970; FPR 1.2%",
            "project_alignment": "Direct grounding for the supervised tree ensemble and metadata-window design",
            "source": "Local project PDF",
        },
        {
            "paper": "Advance Approach for Detection of DNS Tunneling Attack from Network Packets Using Deep Learning Algorithms",
            "year": 2021,
            "architecture": "Bidirectional LSTM",
            "configuration": "Embedding input; bidirectional LSTM with 64 tanh units; 80/20 experiment reported",
            "dataset_scope": "Generated benign/tunneling packet dataset using dnscat2 and packet-query payloads",
            "reported_metrics": "LSTM-family precision 0.99, recall 0.98, F1 0.98 on the 80/20 split; overall NLP feature-extraction test accuracy 98.42%",
            "project_alignment": "Direct family/architecture grounding; project uses ordered metadata channels and completed-window inference rather than raw query text",
            "source": "https://doi.org/10.14201/ADCAIJ2021103241266",
        },
        {
            "paper": "GraphTunnel: Robust DNS Tunnel Detection Based on DNS Recursive Resolution Graph",
            "year": 2024,
            "architecture": "GraphSAGE",
            "configuration": "K=20 recursive paths; 60/40 split; batch 64; learning rate 0.005",
            "dataset_scope": "Known and unknown tunneling tools plus wildcard-DNS robustness traffic",
            "reported_metrics": "100% non-wildcard accuracy; wildcard F1 99.78%; tool accuracy above 98.57%",
            "project_alignment": "Direct structural grounding; project substitutes a documented client-resolver-domain proxy graph",
            "source": "Local project PDF",
        },
        {
            "paper": "KRTunnel: DNS Channel Detector for Mobile Devices",
            "year": 2022,
            "architecture": "Isolation Forest",
            "configuration": "20,000 Android DNS samples; 10,000 train and 10,000 test; request/response metadata features",
            "dataset_scope": "Balanced Android benign and DNS-tunnel traffic",
            "reported_metrics": "Precision 0.989; recall 0.973; accuracy 0.981; F1 0.978",
            "project_alignment": "Direct anomaly-family grounding; project strengthens the protocol by fitting on benign Train only and selecting thresholds on Validation",
            "source": "https://doi.org/10.1016/j.cose.2022.102818",
        },
        {
            "paper": "Identifying Malicious DNS Tunnel Tools from DoH Traffic Using Hierarchical Machine Learning Classification",
            "year": 2021,
            "architecture": "XGBoost / LightGBM / CatBoost",
            "configuration": "139 tuned candidates; stratified 10-fold evaluation with 9:1 train/test per fold",
            "dataset_scope": "CIRA-CIC-DoHBrw-2020",
            "reported_metrics": "Accuracy 99.81% DoH filtering; 99.99% suspicious DoH detection; 97.22% tool identification",
            "project_alignment": "Corroborates nonlinear tree ensembles and emphasizes dataset/environment limitations; not a fifth project model",
            "source": "Local project PDF",
        },
    ]
    pd.DataFrame(literature_rows).to_csv(OUTPUT_DIR / "literature_grounding_matrix_revised.csv", index=False, encoding="utf-8-sig")

    exclusions = []
    for split in ("train", "validation", "test"):
        path = SOURCE_DIR / "representations" / split / f"{split}_excluded_small_groups.csv"
        frame = pd.read_csv(path, low_memory=False)
        exclusions.append({"split": split, "excluded_groups": int(len(frame)), "excluded_transactions": int(frame["transactions"].sum())})
    pd.DataFrame(exclusions).to_csv(OUTPUT_DIR / "small_group_exclusion_summary.csv", index=False, encoding="utf-8-sig")
    full_tabular = SOURCE_DIR / "representations" / "train" / "train_tabular_raw.csv"
    core_tabular = ROOT / "comprehensive_corrections" / "representations_core42" / "train_tabular_core42_raw.csv"
    cost_summary = {
        "train_transactions": 1_401_186,
        "production_windows": 10_971,
        "transaction_to_production_row_reduction_factor": round(1_401_186 / 10_971, 1),
        "diagnostic_windows": 2_362,
        "transaction_to_diagnostic_row_reduction_factor": round(1_401_186 / 2_362, 1),
        "audited_tabular_features": 52,
        "primary_core_features": 42,
        "column_reduction_percent": round((52 - 42) / 52 * 100, 1),
        "production_feature_cells_before": 10_971 * 52,
        "production_feature_cells_after": 10_971 * 42,
        "production_feature_cells_removed": 10_971 * 10,
        "full_train_tabular_raw_mb": round(full_tabular.stat().st_size / 1024**2, 2),
        "core_train_tabular_raw_mb": round(core_tabular.stat().st_size / 1024**2, 2),
        "train_tabular_file_reduction_percent": round((1 - core_tabular.stat().st_size / full_tabular.stat().st_size) * 100, 1),
    }
    pd.DataFrame([cost_summary]).to_csv(OUTPUT_DIR / "computational_cost_reduction.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"literature_rows": len(literature_rows), "protocol_updated": True, "exclusions": exclusions, "cost_summary": cost_summary}, indent=2))


if __name__ == "__main__":
    main()
