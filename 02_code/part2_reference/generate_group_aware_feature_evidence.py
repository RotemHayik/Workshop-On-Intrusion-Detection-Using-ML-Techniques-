from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
TABULAR = ROOT / "chapter6_preparation" / "representations" / "train" / "train_tabular_raw.csv"
MANIFEST = ROOT / "chapter6_preparation" / "representations" / "train" / "train_sample_manifest.csv"
SCHEMA = ROOT / "chapter5_outputs" / "Appendix_D_Unified_Feature_Schema.csv"
OUTPUT_DIR = ROOT / "comprehensive_corrections" / "group_aware_evidence"
SEED = 20260819
BOOTSTRAPS = 2000


def source_short(value: object) -> str:
    text = str(value).lower()
    if "nitsan" in text:
        return "Nitsan"
    if "itamar" in text:
        return "Itamar"
    if "rotem" in text:
        return "Rotem"
    if "ctu" in text:
        return "CTU Normal"
    return str(value)


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return math.nan
    right_sorted = np.sort(right)
    less = np.searchsorted(right_sorted, left, side="left")
    greater = right_sorted.size - np.searchsorted(right_sorted, left, side="right")
    return float(np.mean((less - greater) / right_sorted.size))


def mann_whitney_normal(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    n1, n2 = left.size, right.size
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    combined = np.concatenate([left, right])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    n = n1 + n2
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    variance = n1 * n2 / 12 * ((n + 1) - tie_sum / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        return u1, math.nan
    mean_u = n1 * n2 / 2
    correction = 0.5 * np.sign(u1 - mean_u)
    z = (u1 - mean_u - correction) / math.sqrt(variance)
    return u1, math.erfc(abs(z) / math.sqrt(2))


def holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        adjusted = min(1.0, float(value) * (count - rank))
        running = max(running, adjusted)
        result.loc[index] = running
    return result


def effect_magnitude(delta: float) -> str:
    if not math.isfinite(delta):
        return "Unavailable"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "Negligible"
    if magnitude < 0.33:
        return "Small"
    if magnitude < 0.474:
        return "Medium"
    return "Large"


def bootstrap_delta_ci(left: np.ndarray, right: np.ndarray, seed: int) -> tuple[float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        left_sample = rng.choice(left, size=left.size, replace=True)
        right_sample = rng.choice(right, size=right.size, replace=True)
        estimates[index] = cliffs_delta(left_sample, right_sample)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def analyze_scope(group_frame: pd.DataFrame, features: list[str], scope: str) -> pd.DataFrame:
    scoped = group_frame if scope == "Pooled" else group_frame.loc[group_frame["source_short"] == "Nitsan"]
    malicious = scoped.loc[scoped["label"] == 1]
    benign = scoped.loc[scoped["label"] == 0]
    rows: list[dict] = []
    for feature_index, feature in enumerate(features):
        left = pd.to_numeric(malicious[feature], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(benign[feature], errors="coerce").to_numpy(dtype=float)
        delta = cliffs_delta(left, right)
        u_statistic, p_value = mann_whitney_normal(left, right)
        ci_low, ci_high = bootstrap_delta_ci(left, right, SEED + feature_index + (1000 if scope != "Pooled" else 0))
        rows.append(
            {
                "scope": scope,
                "feature": feature,
                "malicious_groups": int(np.isfinite(left).sum()),
                "benign_groups": int(np.isfinite(right).sum()),
                "malicious_group_median": float(np.nanmedian(left)),
                "benign_group_median": float(np.nanmedian(right)),
                "cliffs_delta": delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "effect_magnitude": effect_magnitude(delta),
                "mann_whitney_u": u_statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_52"] = holm_adjust(result["p_value"])
    result["significant_holm_005"] = result["p_holm_52"] < 0.05
    return result


def decision_for(row: pd.Series) -> str:
    pooled_supported = bool(row["pooled_significant_holm_005"]) and abs(float(row["pooled_cliffs_delta"])) >= 0.147
    nitsan_supported = bool(row["nitsan_significant_holm_005"]) and abs(float(row["nitsan_cliffs_delta"])) >= 0.147
    sign_consistent = np.sign(float(row["pooled_cliffs_delta"])) == np.sign(float(row["nitsan_cliffs_delta"]))
    if pooled_supported and nitsan_supported and sign_consistent:
        return "Retain: group-aware pooled and within-Nitsan evidence"
    if pooled_supported or nitsan_supported:
        return "Retain for ablation: source-sensitive or scope-dependent evidence"
    return "Do not claim as independently justified; supporting/ablation only"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def create_effect_figure(evidence: pd.DataFrame) -> None:
    selected = [
        ("query_name_len_mean", "Mean query-name length"),
        ("query_entropy_mean", "Mean query entropy"),
        ("longest_subdomain_entropy_mean", "Mean subdomain entropy"),
        ("transaction_bytes_mean", "Mean transaction bytes"),
        ("top_base_domain_share", "Top base-domain share"),
        ("unique_base_domain_ratio", "Unique base-domain ratio"),
        ("response_time_std", "Response-time variability"),
        ("iat_mean", "Mean interarrival time"),
        ("qtype_a_ratio", "A-query ratio"),
        ("repeated_query_ratio", "Repeated-query ratio"),
    ]
    lookup = evidence.set_index("feature")
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, blue, orange, gray, dark = "#17324D", "#39749B", "#E29B33", "#D8E0E7", "#1D2730"
    draw.text((70, 45), "Recording-Level Effect Sizes for Representative Core Signals", font=font(40, True), fill=navy)
    draw.text((70, 100), "Cliff's delta from split_group_id medians; whiskers are 95% recording-bootstrap intervals", font=font(23), fill="#52606D")
    plot_left, plot_right, plot_top, plot_bottom = 690, 1700, 190, 930
    for tick in np.linspace(-1, 1, 9):
        x = plot_left + (plot_right - plot_left) * (tick + 1) / 2
        draw.line((x, plot_top, x, plot_bottom), fill=navy if abs(tick) < 1e-9 else gray, width=3 if abs(tick) < 1e-9 else 2)
        label = f"{tick:.2f}" if tick not in (-1, 0, 1) else f"{tick:.0f}"
        draw.text((x, plot_bottom + 18), label, font=font(19), fill="#64707C", anchor="ma")
    row_height = (plot_bottom - plot_top) / len(selected)
    for index, (feature, label) in enumerate(selected):
        row = lookup.loc[feature]
        y = plot_top + (index + 0.5) * row_height
        draw.text((plot_left - 28, y), label, font=font(22), fill=dark, anchor="rm")
        for offset, prefix, color in [(-10, "pooled", blue), (10, "nitsan", orange)]:
            value = float(row[f"{prefix}_cliffs_delta"])
            low = float(row[f"{prefix}_delta_ci_low"])
            high = float(row[f"{prefix}_delta_ci_high"])
            x_value = plot_left + (plot_right - plot_left) * (value + 1) / 2
            x_low = plot_left + (plot_right - plot_left) * (low + 1) / 2
            x_high = plot_left + (plot_right - plot_left) * (high + 1) / 2
            draw.line((x_low, y + offset, x_high, y + offset), fill=color, width=5)
            draw.line((x_low, y + offset - 7, x_low, y + offset + 7), fill=color, width=3)
            draw.line((x_high, y + offset - 7, x_high, y + offset + 7), fill=color, width=3)
            draw.ellipse((x_value - 8, y + offset - 8, x_value + 8, y + offset + 8), fill=color, outline="white", width=2)
    draw.rectangle((700, 980, 730, 1004), fill=blue)
    draw.text((742, 978), "Pooled sources", font=font(21), fill=dark)
    draw.rectangle((950, 980, 980, 1004), fill=orange)
    draw.text((992, 978), "Within Nitsan", font=font(21), fill=dark)
    draw.text((1260, 978), "Positive = larger in malicious", font=font(20), fill="#52606D")
    image.save(OUTPUT_DIR / "Figure_3_2_Group_Aware_Effect_Sizes.png", dpi=(220, 220))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabular = pd.read_csv(TABULAR, low_memory=False)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    schema = pd.read_csv(SCHEMA, low_memory=False)
    features = schema["feature"].astype(str).tolist()
    expected = [column for column in tabular.columns if column not in {"sample_id", "label"}]
    if features != expected:
        raise RuntimeError("Appendix D feature order does not match the production tabular representation")
    merged = tabular.merge(
        manifest[["sample_id", "split_group_id", "source_name", "traffic_label"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(tabular):
        raise RuntimeError("Sample manifest and tabular representation are not aligned")
    merged["source_short"] = merged["source_name"].map(source_short)
    merged["label"] = pd.to_numeric(merged["label"], errors="raise").astype(int)
    group_frame = (
        merged.groupby("split_group_id", sort=False)
        .agg(
            label=("label", "first"),
            source_short=("source_short", "first"),
            windows=("sample_id", "size"),
            **{feature: (feature, "median") for feature in features},
        )
        .reset_index()
    )
    if group_frame["split_group_id"].duplicated().any() or len(group_frame) != 130:
        raise RuntimeError("Expected exactly 130 represented Train recording groups")

    pooled = analyze_scope(group_frame, features, "Pooled")
    nitsan = analyze_scope(group_frame, features, "Within Nitsan")
    long_form = pd.concat([pooled, nitsan], ignore_index=True)
    long_form.to_csv(OUTPUT_DIR / "Appendix_C_Group_Aware_52_Feature_Tests_Long.csv", index=False, encoding="utf-8-sig")

    pooled_wide = pooled.add_prefix("pooled_").rename(columns={"pooled_feature": "feature"})
    nitsan_wide = nitsan.add_prefix("nitsan_").rename(columns={"nitsan_feature": "feature"})
    evidence = schema.merge(pooled_wide, on="feature", validate="one_to_one").merge(nitsan_wide, on="feature", validate="one_to_one")
    evidence["direction_consistent"] = np.sign(evidence["pooled_cliffs_delta"]) == np.sign(evidence["nitsan_cliffs_delta"])
    evidence["final_empirical_decision"] = evidence.apply(decision_for, axis=1)
    evidence.to_csv(OUTPUT_DIR / "Appendix_C_Group_Aware_52_Feature_Evidence.csv", index=False, encoding="utf-8-sig")
    create_effect_figure(evidence)
    group_frame.to_csv(OUTPUT_DIR / "Train_Recording_Level_Medians.csv", index=False, encoding="utf-8-sig")

    core_features = evidence.loc[
        evidence["final_empirical_decision"] != "Do not claim as independently justified; supporting/ablation only",
        "feature",
    ].astype(str).tolist()
    core_schema = evidence.loc[evidence["feature"].isin(core_features)].copy()
    core_schema.insert(0, "core_feature_number", np.arange(1, len(core_schema) + 1))
    core_schema.to_csv(OUTPUT_DIR / "Appendix_D_Core_42_Feature_Schema.csv", index=False, encoding="utf-8-sig")
    representation_dir = ROOT / "comprehensive_corrections" / "representations_core42"
    representation_dir.mkdir(parents=True, exist_ok=True)
    core_representation_rows: dict[str, int] = {}
    for split in ("train", "validation", "test", "external_test"):
        split_dir = ROOT / "chapter6_preparation" / "representations" / split
        for variant in ("raw", "scaled"):
            source_path = split_dir / f"{split}_tabular_{variant}.csv"
            frame = pd.read_csv(source_path, low_memory=False)
            selected = frame[["sample_id", "label", *core_features]].copy()
            selected.to_csv(
                representation_dir / f"{split}_tabular_core42_{variant}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            core_representation_rows[f"{split}_{variant}"] = int(len(selected))

    summary = {
        "train_windows": int(len(merged)),
        "represented_train_groups": int(len(group_frame)),
        "pooled_benign_groups": int(((group_frame["label"] == 0)).sum()),
        "pooled_malicious_groups": int(((group_frame["label"] == 1)).sum()),
        "nitsan_benign_groups": int(((group_frame["label"] == 0) & (group_frame["source_short"] == "Nitsan")).sum()),
        "nitsan_malicious_groups": int(((group_frame["label"] == 1) & (group_frame["source_short"] == "Nitsan")).sum()),
        "features": len(features),
        "pooled_holm_significant": int(pooled["significant_holm_005"].sum()),
        "nitsan_holm_significant": int(nitsan["significant_holm_005"].sum()),
        "retained_both_consistent": int((evidence["final_empirical_decision"] == "Retain: group-aware pooled and within-Nitsan evidence").sum()),
        "retained_ablation": int((evidence["final_empirical_decision"] == "Retain for ablation: source-sensitive or scope-dependent evidence").sum()),
        "supporting_only": int((evidence["final_empirical_decision"] == "Do not claim as independently justified; supporting/ablation only").sum()),
        "core_feature_count": len(core_features),
        "core_representation_rows": core_representation_rows,
        "bootstrap_replicates": BOOTSTRAPS,
        "multiple_testing": "Holm family-wise correction across 52 features, separately for pooled and within-Nitsan tests",
        "independent_unit": "split_group_id recording median",
    }
    (OUTPUT_DIR / "group_aware_evidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
