from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "chapter4_outputs" / "train_window_features.csv"
OUTPUT_DIR = ROOT / "chapter5_outputs"
META_COLUMNS = ["split_group_id", "traffic_label", "source_name", "window_number", "window_tx_count"]

ROTEM_SOURCE = "dns-tunnel-dataset-master - origin - Rotem"
STRATA = [
    ("Nitsan", "benign", "A", "Nitsan benign", "#3D7EA6"),
    ("CTU_Normal_Benign", "benign", "B", "CTU benign", "#65A88D"),
    ("Nitsan", "malicious", "C", "Nitsan malicious", "#E09B3E"),
    ("Itamar", "malicious", "D", "Itamar malicious", "#C95A50"),
    (ROTEM_SOURCE, "malicious", "E", "Rotem malicious", "#7C5AA6"),
]

PLOT_FEATURES = [
    ("query_name_len_mean", "Mean query-name length", "log1p", "log(1 + characters)"),
    ("query_entropy_mean", "Mean query entropy", "identity", "bits / character"),
    ("transaction_bytes_mean", "Mean transaction bytes", "log1p", "log(1 + bytes)"),
    ("unique_base_domain_ratio", "Unique base-domain ratio", "identity", "ratio"),
    ("qtype_a_ratio", "A-query ratio", "identity", "ratio"),
    ("response_time_std", "Response-time variability", "log1p", "log(1 + seconds)"),
]


def feature_family(feature: str) -> str:
    if "query_name_len" in feature or "subdomain_len" in feature or "num_labels" in feature or "max_label_len" in feature:
        return "Lexical structure"
    if "entropy" in feature:
        return "Encoding / entropy"
    if "consonant" in feature:
        return "Lexical encoding"
    if "bytes" in feature or "payload" in feature:
        return "Transaction volume"
    if "response_time" in feature or "response_missing" in feature:
        return "Response / timing"
    if feature.startswith("iat_"):
        return "Temporal cadence"
    if "base_domain" in feature or "unique_subdomain" in feature or "repeated_query" in feature:
        return "Domain recurrence"
    if "jaro_winkler" in feature:
        return "Sequence similarity"
    if feature.startswith("qtype_"):
        return "DNS type distribution"
    if feature.startswith("rcode_"):
        return "DNS response distribution"
    return "Other"


def feature_unit(feature: str) -> str:
    if "len" in feature and "num_labels" not in feature:
        return "characters"
    if "num_labels" in feature:
        return "labels"
    if "entropy" in feature:
        return "bits/character"
    if "bytes" in feature or "payload" in feature:
        return "bytes"
    if feature.startswith("iat_") or "response_time" in feature:
        return "seconds" if not feature.endswith("_cv") else "unitless"
    if feature.endswith("_ratio") or "share" in feature or "jaro_winkler" in feature:
        return "ratio [0,1]"
    return "count / unitless"


def definition(feature: str) -> str:
    special = {
        "top_base_domain_share": "Maximum base-domain frequency divided by window size",
        "unique_base_domain_ratio": "Unique base domains divided by window size",
        "unique_subdomain_ratio": "Unique subdomains divided by window size",
        "repeated_query_ratio": "1 - unique normalized query names / window size",
        "iat_cv": "Inter-arrival-time standard deviation divided by mean",
        "adjacent_subdomain_jaro_winkler_mean": "Mean Jaro-Winkler similarity of sampled adjacent subdomains",
        "response_missing_ratio": "Transactions without a matched response divided by window size",
    }
    if feature in special:
        return special[feature]
    if feature.endswith("_mean"):
        return "Arithmetic mean within the recording-safe window"
    if feature.endswith("_std"):
        return "Population standard deviation within the window"
    if feature.endswith("_median"):
        return "Median within the window"
    if feature.endswith("_p95"):
        return "95th percentile within the window"
    if feature.endswith("_ratio"):
        return "Matching transactions divided by window size"
    return "Window-level scalar derived from normalized transactions"


def is_bounded(feature: str) -> bool:
    return feature.endswith("_ratio") or "share" in feature or "jaro_winkler" in feature


def transform_rule(feature: str) -> str:
    if is_bounded(feature):
        return "clip[0,1]; identity"
    if "entropy" in feature:
        return "robust: (x - median_train) / IQR_train"
    return "log1p then robust: (log1p(x) - median_train) / IQR_train"


def missing_rule(feature: str) -> str:
    if "response_time" in feature:
        return "Train median after transform + response-missing indicator"
    if feature.startswith("rcode_"):
        return "NULL/OTHER ratio retained; remaining missing values use Train median"
    return "Train median after transform if unavailable"


def empirical_status(feature: str) -> str:
    if feature.startswith("qtype_"):
        return "Modify: source-sensitive; use normalized ratios and ablation"
    if "response_time" in feature or feature.startswith("iat_"):
        return "Modify: source-sensitive; source-stratified validation"
    if feature in {"request_payload_mean", "subdomain_len_mean", "query_name_len_mean"}:
        return "Keep as a correlated behavioral family; regularize/consolidate"
    if feature.startswith("rcode_") or feature == "response_missing_ratio":
        return "Modify: correlated response-availability family"
    return "Keep subject to grouped validation"


def transformed_values(values: pd.Series, rule: str) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if rule.startswith("log1p"):
        array = np.log1p(np.clip(array, 0, None))
    elif rule.startswith("clip"):
        array = np.clip(array, 0, 1)
    return array


def build_schema(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [column for column in frame.columns if column not in META_COLUMNS]
    schema_rows = []
    scaling_rows = []
    for feature in features:
        rule = transform_rule(feature)
        values = transformed_values(frame[feature], rule)
        valid = values[np.isfinite(values)]
        median = float(np.median(valid)) if valid.size else 0.0
        q1 = float(np.quantile(valid, 0.25)) if valid.size else 0.0
        q3 = float(np.quantile(valid, 0.75)) if valid.size else 0.0
        iqr = q3 - q1
        scale = iqr if iqr > 1e-9 else 1.0
        availability = "Partial by source/response pairing" if ("response_time" in feature or feature.startswith("rcode_") or feature == "response_missing_ratio") else "Available after unified parsing"
        schema_rows.append({
            "feature": feature,
            "family": feature_family(feature),
            "definition": definition(feature),
            "unit": feature_unit(feature),
            "availability": availability,
            "train_transformation": rule,
            "missing_value_rule": missing_rule(feature),
            "empirical_status": empirical_status(feature),
        })
        scaling_rows.append({
            "feature": feature,
            "transformation": rule,
            "train_center": median,
            "train_iqr": iqr,
            "applied_scale": scale,
            "fit_partition": "Train only",
        })
    schema = pd.DataFrame(schema_rows)
    scaling = pd.DataFrame(scaling_rows)
    schema.to_csv(OUTPUT_DIR / "Appendix_D_Unified_Feature_Schema.csv", index=False, encoding="utf-8-sig")
    scaling.to_csv(OUTPUT_DIR / "Train_Only_Scaling_Parameters.csv", index=False, encoding="utf-8-sig")
    return schema, scaling


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(left[np.isfinite(left)])
    right = np.sort(right[np.isfinite(right)])
    if not len(left) or not len(right):
        return math.nan
    combined = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, combined, side="right") / len(left)
    right_cdf = np.searchsorted(right, combined, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def robust_median_shift(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        return math.nan
    scale = float(np.quantile(left, 0.75) - np.quantile(left, 0.25))
    if scale <= 1e-9:
        scale = max(float(np.std(left)), 1e-9)
    return float(abs(np.median(right) - np.median(left)) / scale)


def subset(frame: pd.DataFrame, source: str, label: str) -> pd.DataFrame:
    return frame[(frame["source_name"].astype(str) == source) & (frame["traffic_label"].astype(str).str.lower() == label)]


def distribution_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for source, label, code, display, _ in STRATA:
        part = subset(frame, source, label)
        for feature, _, plot_transform, _ in PLOT_FEATURES:
            values = transformed_values(part[feature], plot_transform)
            values = values[np.isfinite(values)]
            summary_rows.append({
                "stratum_code": code,
                "stratum": display,
                "source_name": source,
                "traffic_label": label,
                "feature": feature,
                "windows": len(values),
                "groups": int(part["split_group_id"].nunique()),
                "p05": float(np.quantile(values, 0.05)),
                "q1": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q3": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "Source_Stratum_Distribution_Summary.csv", index=False, encoding="utf-8-sig")

    shift_rows = []
    for feature, _, plot_transform, _ in PLOT_FEATURES:
        nitsan_benign = transformed_values(subset(frame, "Nitsan", "benign")[feature], plot_transform)
        ctu_benign = transformed_values(subset(frame, "CTU_Normal_Benign", "benign")[feature], plot_transform)
        malicious = {
            "Nitsan": transformed_values(subset(frame, "Nitsan", "malicious")[feature], plot_transform),
            "Itamar": transformed_values(subset(frame, "Itamar", "malicious")[feature], plot_transform),
            "Rotem": transformed_values(subset(frame, ROTEM_SOURCE, "malicious")[feature], plot_transform),
        }
        malicious_pairs = []
        keys = list(malicious)
        for left_index in range(len(keys)):
            for right_index in range(left_index + 1, len(keys)):
                left, right = keys[left_index], keys[right_index]
                malicious_pairs.append((
                    f"{left} vs {right}",
                    ks_statistic(malicious[left], malicious[right]),
                    robust_median_shift(malicious[left], malicious[right]),
                ))
        max_pair = max(malicious_pairs, key=lambda value: value[1])
        shift_rows.append({
            "feature": feature,
            "benign_comparison": "Nitsan benign vs CTU benign",
            "benign_ks": ks_statistic(nitsan_benign, ctu_benign),
            "benign_robust_median_shift": robust_median_shift(nitsan_benign, ctu_benign),
            "largest_malicious_pair": max_pair[0],
            "largest_malicious_ks": max_pair[1],
            "largest_malicious_robust_median_shift": max_pair[2],
        })
    shifts = pd.DataFrame(shift_rows).sort_values("benign_ks", ascending=False)
    shifts.to_csv(OUTPUT_DIR / "Distribution_Shift_Metrics.csv", index=False, encoding="utf-8-sig")
    return summary, shifts


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def draw_panel(draw: ImageDraw.ImageDraw, panel: tuple[int, int, int, int], title: str, unit: str, rows: pd.DataFrame, colors: dict[str, str]) -> None:
    left, top, right, bottom = panel
    draw.rounded_rectangle(panel, radius=18, fill="#FAFBFC", outline="#CFD8E1", width=2)
    draw.text((left + 24, top + 18), title, font=font(27, True), fill="#17324D")
    draw.text((left + 24, top + 55), unit, font=font(20), fill="#66727E")
    plot_left, plot_right = left + 95, right - 30
    plot_top, plot_bottom = top + 105, bottom - 55
    minimum = float(rows["p05"].min())
    maximum = float(rows["p95"].max())
    if unit == "ratio":
        minimum, maximum = 0.0, 1.0
    if maximum <= minimum:
        maximum = minimum + 1.0
    if unit != "ratio":
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding
    for tick in np.linspace(minimum, maximum, 4):
        y = plot_bottom - (tick - minimum) / (maximum - minimum) * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#E2E7EC", width=2)
        label = f"{tick:.2f}" if abs(tick) < 10 else f"{tick:.0f}"
        draw.text((left + 18, y - 11), label, font=font(17), fill="#6B7580")
    x_positions = np.linspace(plot_left + 45, plot_right - 45, len(rows))
    for x, row in zip(x_positions, rows.itertuples(index=False)):
        def y_value(value: float) -> float:
            return plot_bottom - (value - minimum) / (maximum - minimum) * (plot_bottom - plot_top)
        p05, q1, median, q3, p95 = map(y_value, [row.p05, row.q1, row.median, row.q3, row.p95])
        color = colors[row.stratum_code]
        draw.line((x, p95, x, p05), fill=color, width=4)
        draw.line((x - 12, p95, x + 12, p95), fill=color, width=4)
        draw.line((x - 12, p05, x + 12, p05), fill=color, width=4)
        draw.rounded_rectangle((x - 25, q3, x + 25, q1), radius=6, fill=color, outline="#FFFFFF", width=2)
        draw.line((x - 24, median, x + 24, median), fill="#FFFFFF", width=4)
        draw.text((x - 8, plot_bottom + 16), row.stratum_code, font=font(21, True), fill="#24313C")


def create_figure(summary: pd.DataFrame) -> None:
    width, height = 1900, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "Train-Only Cross-Source Distribution Shift", font=font(43, True), fill="#17324D")
    draw.text((70, 102), "Recording-safe window features; whiskers show 5th-95th percentiles", font=font(26), fill="#5B6875")
    colors = {code: color for _, _, code, _, color in STRATA}
    legend_x = 70
    for source, label, code, display, color in STRATA:
        row = summary[summary["stratum_code"] == code].iloc[0]
        draw.rounded_rectangle((legend_x, 150, legend_x + 28, 178), radius=5, fill=color)
        label_text = f"{code}: {display} (n={int(row.windows)}, g={int(row.groups)})"
        draw.text((legend_x + 38, 148), label_text, font=font(19), fill="#26333E")
        legend_x += 350 if code != "C" else 375
    margin_x, gap_x = 60, 28
    panel_w = (width - 2 * margin_x - 2 * gap_x) // 3
    panel_h = 490
    top_start = 220
    for index, (feature, title, _, unit) in enumerate(PLOT_FEATURES):
        row_index, col_index = divmod(index, 3)
        left = margin_x + col_index * (panel_w + gap_x)
        top = top_start + row_index * (panel_h + 28)
        rows = summary[summary["feature"] == feature].set_index("stratum_code").loc[["A", "B", "C", "D", "E"]].reset_index()
        draw_panel(draw, (left, top, left + panel_w, top + panel_h), title, unit, rows, colors)
    image.save(OUTPUT_DIR / "Figure_5_1_Cross_Source_Distribution_Shift.png", dpi=(220, 220))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_CSV, low_memory=False)
    schema, scaling = build_schema(frame)
    summary, shifts = distribution_analysis(frame)
    create_figure(summary)
    result = {
        "windows": int(len(frame)),
        "groups": int(frame["split_group_id"].nunique()),
        "features": int(len(schema)),
        "source_strata": int(len(STRATA)),
        "scaling_parameters": int(len(scaling)),
        "highest_benign_shift_feature": str(shifts.iloc[0]["feature"]),
        "highest_benign_ks": float(shifts.iloc[0]["benign_ks"]),
        "shift_metrics": shifts.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "chapter5_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
