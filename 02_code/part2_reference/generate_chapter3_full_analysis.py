from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path(r"G:\My Drive\Workshop Files\Rotem\train_transactions.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "chapter3_full_outputs"
CHUNK_SIZE = 120_000
ROW_SAMPLE_MODULUS = 20  # deterministic 5% sample for quantiles/effect sizes

CORE_NUMERIC = [
    "query_name_len",
    "subdomain_len",
    "num_labels",
    "max_label_len",
    "query_entropy",
    "longest_subdomain_entropy",
    "long_consonant_string_count",
    "req_resp_time_diff",
    "total_transaction_bytes",
    "mean_ttl",
]

CORRELATION_FEATURES = [
    "malicious_label",
    "query_name_len",
    "subdomain_len",
    "num_labels",
    "max_label_len",
    "query_entropy",
    "longest_subdomain_entropy",
    "long_consonant_string_count",
    "total_transaction_bytes",
]

MISSINGNESS_FIELDS = [
    "query_name",
    "base_domain",
    "subdomain",
    "query_type",
    "dns_rcode",
    "mean_ttl",
    "req_resp_time_diff",
    "total_transaction_bytes",
    "query_name_len",
    "query_entropy",
]

LOW_CARDINALITY_FIELDS = [
    "is_dns",
    "is_doh_traffic",
    "application_hint",
    "transport_protocol",
    "ip_version",
    "query_class",
    "curation_decision",
    "is_response_missing",
    "query_type",
    "dns_rcode",
]

READ_COLUMNS = list(
    dict.fromkeys(
        [
            "transaction_id",
            "source_name",
            "traffic_label",
            "split_group_id",
            "client_ip",
            "base_domain",
            "subdomain",
            "query_name",
            "request_timestamp",
            *CORE_NUMERIC,
            *LOW_CARDINALITY_FIELDS,
        ]
    )
)


def short_source(value: str) -> str:
    lowered = str(value).lower()
    if "ctu" in lowered:
        return "CTU Normal"
    if "nitsan" in lowered:
        return "Nitsan"
    if "itamar" in lowered:
        return "Itamar"
    if "rotem" in lowered:
        return "Rotem"
    return str(value)


def stratum(source: str, label: str) -> str:
    return f"{short_source(source)} {str(label).title()}"


def stable_order(value: str) -> int:
    return int.from_bytes(hashlib.sha1(value.encode("utf-8", errors="ignore")).digest()[:8], "big")


def is_absent(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        missing = missing | series.astype("string").str.strip().eq("").fillna(True)
    return missing


def update_pairwise_stats(pair_stats: dict, numeric: pd.DataFrame) -> None:
    arrays = {feature: numeric[feature].to_numpy(dtype=np.float64) for feature in CORRELATION_FEATURES}
    for (left, right), stats in pair_stats.items():
        x = arrays[left]
        y = arrays[right]
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            continue
        xv = x[valid]
        yv = y[valid]
        stats["n"] += int(xv.size)
        stats["sum_x"] += float(xv.sum())
        stats["sum_y"] += float(yv.sum())
        stats["sum_xx"] += float(np.dot(xv, xv))
        stats["sum_yy"] += float(np.dot(yv, yv))
        stats["sum_xy"] += float(np.dot(xv, yv))


def finalize_correlations(pair_stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(np.nan, index=CORRELATION_FEATURES, columns=CORRELATION_FEATURES, dtype=float)
    counts = pd.DataFrame(0, index=CORRELATION_FEATURES, columns=CORRELATION_FEATURES, dtype=int)
    for (left, right), stats in pair_stats.items():
        n = stats["n"]
        numerator = n * stats["sum_xy"] - stats["sum_x"] * stats["sum_y"]
        denominator_x = n * stats["sum_xx"] - stats["sum_x"] ** 2
        denominator_y = n * stats["sum_yy"] - stats["sum_y"] ** 2
        denominator = math.sqrt(max(0.0, denominator_x) * max(0.0, denominator_y))
        correlation = numerator / denominator if denominator else math.nan
        matrix.loc[left, right] = correlation
        matrix.loc[right, left] = correlation
        counts.loc[left, right] = n
        counts.loc[right, left] = n
    return matrix, counts


def cliffs_delta(malicious: np.ndarray, benign: np.ndarray, cap: int = 20_000) -> float:
    malicious = malicious[np.isfinite(malicious)]
    benign = benign[np.isfinite(benign)]
    rng = np.random.default_rng(20260819)
    if malicious.size > cap:
        malicious = rng.choice(malicious, size=cap, replace=False)
    if benign.size > cap:
        benign = rng.choice(benign, size=cap, replace=False)
    if malicious.size == 0 or benign.size == 0:
        return math.nan
    benign_sorted = np.sort(benign)
    less = np.searchsorted(benign_sorted, malicious, side="left")
    greater = benign_sorted.size - np.searchsorted(benign_sorted, malicious, side="right")
    return float(np.mean((less - greater) / benign_sorted.size))


def mann_whitney_normal_approx(malicious: np.ndarray, benign: np.ndarray, cap: int = 20_000) -> tuple[float, float]:
    malicious = malicious[np.isfinite(malicious)]
    benign = benign[np.isfinite(benign)]
    rng = np.random.default_rng(20260819)
    if malicious.size > cap:
        malicious = rng.choice(malicious, size=cap, replace=False)
    if benign.size > cap:
        benign = rng.choice(benign, size=cap, replace=False)
    n1, n2 = malicious.size, benign.size
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan
    combined = np.concatenate([malicious, benign])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    n = n1 + n2
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(tie_counts**3 - tie_counts))
    variance = n1 * n2 / 12 * ((n + 1) - tie_sum / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        return u1, math.nan
    mean_u = n1 * n2 / 2
    z = (u1 - mean_u) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return u1, p_value


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


def quantile_summary(frame: pd.DataFrame, group_column: str, features: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for group_value, group in frame.groupby(group_column, dropna=False):
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    group_column: group_value,
                    "feature": feature,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "missing_rate": float(1 - values.size / len(group)),
                }
            )
    return pd.DataFrame(rows)


def character_ratio(value: object, allowed: set[str]) -> float:
    if value is None or pd.isna(value):
        return math.nan
    text = "".join(character for character in str(value) if character != ".")
    if not text:
        return 0.0
    return sum(character in allowed for character in text) / len(text)


def first_pass() -> dict:
    class_counts: Counter = Counter()
    source_counts: Counter = Counter()
    source_groups: defaultdict[str, set[str]] = defaultdict(set)
    missing_counts: Counter = Counter()
    stratum_counts: Counter = Counter()
    low_card_counts: dict[str, Counter] = {field: Counter() for field in LOW_CARDINALITY_FIELDS}
    category_counts: dict[str, Counter] = {
        "query_type_pooled_benign": Counter(),
        "query_type_pooled_malicious": Counter(),
        "query_type_nitsan_benign": Counter(),
        "query_type_nitsan_malicious": Counter(),
        "dns_rcode_pooled_benign": Counter(),
        "dns_rcode_pooled_malicious": Counter(),
        "dns_rcode_nitsan_benign": Counter(),
        "dns_rcode_nitsan_malicious": Counter(),
    }
    pair_stats = {
        (left, right): {
            "n": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_xx": 0.0,
            "sum_yy": 0.0,
            "sum_xy": 0.0,
        }
        for index, left in enumerate(CORRELATION_FEATURES)
        for right in CORRELATION_FEATURES[index:]
    }
    samples: list[pd.DataFrame] = []
    domain_stats: dict[int, list] = {}
    recording_sizes: Counter = Counter()
    recording_strata: dict[str, str] = {}
    total_rows = 0

    for chunk in pd.read_csv(TRAIN_PATH, usecols=READ_COLUMNS, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["source_short"] = chunk["source_name"].map(short_source)
        chunk["traffic_label"] = chunk["traffic_label"].astype("string").str.lower().str.strip()
        chunk["stratum"] = chunk["source_short"] + " " + chunk["traffic_label"].str.title()
        chunk["malicious_label"] = (chunk["traffic_label"] == "malicious").astype(float)
        total_rows += len(chunk)

        for label, count in chunk["traffic_label"].value_counts(dropna=False).items():
            class_counts[str(label)] += int(count)
        for group_name, count in chunk["stratum"].value_counts(dropna=False).items():
            stratum_counts[str(group_name)] += int(count)
        for (source, label), group in chunk.groupby(["source_short", "traffic_label"], dropna=False):
            key = f"{source}|||{label}"
            source_counts[key] += len(group)
            source_groups[key].update(str(value) for value in group["split_group_id"].dropna().unique())

        for recording, group in chunk.groupby("split_group_id", dropna=False):
            recording_key = str(recording)
            recording_sizes[recording_key] += len(group)
            recording_strata[recording_key] = str(group["stratum"].iloc[0])

        for field in MISSINGNESS_FIELDS:
            absent = is_absent(chunk[field])
            counts = chunk.loc[absent, "stratum"].value_counts(dropna=False)
            for group_name, count in counts.items():
                missing_counts[(str(group_name), field)] += int(count)

        for field in LOW_CARDINALITY_FIELDS:
            normalized = chunk[field].astype("string").fillna("<MISSING>").replace("", "<MISSING>")
            for value, count in normalized.value_counts(dropna=False).items():
                low_card_counts[field][str(value)] += int(count)

        for field in ("query_type", "dns_rcode"):
            values = chunk[field].astype("string").fillna("<MISSING>").replace("", "<MISSING>")
            for label in ("benign", "malicious"):
                label_values = values.loc[chunk["traffic_label"] == label]
                category_counts[f"{field}_pooled_{label}"].update(label_values.value_counts().to_dict())
                nitsan_values = values.loc[(chunk["traffic_label"] == label) & (chunk["source_short"] == "Nitsan")]
                category_counts[f"{field}_nitsan_{label}"].update(nitsan_values.value_counts().to_dict())

        numeric = pd.DataFrame(index=chunk.index)
        for feature in CORRELATION_FEATURES:
            numeric[feature] = pd.to_numeric(chunk[feature], errors="coerce")
        update_pairwise_stats(pair_stats, numeric)

        sample_hash = pd.util.hash_pandas_object(chunk["transaction_id"].astype("string"), index=False).to_numpy(dtype=np.uint64)
        sample_mask = sample_hash % ROW_SAMPLE_MODULUS == 0
        sample_columns = ["source_short", "traffic_label", "stratum", "subdomain", *CORE_NUMERIC]
        samples.append(chunk.loc[sample_mask, sample_columns].copy())

        domain_source = chunk.loc[
            ~is_absent(chunk["base_domain"]) & ~is_absent(chunk["split_group_id"]),
            ["split_group_id", "client_ip", "base_domain", "source_short", "traffic_label", "request_timestamp"],
        ].copy()
        if not domain_source.empty:
            key_columns = domain_source[["split_group_id", "client_ip", "base_domain"]].astype("string").fillna("<MISSING>")
            domain_source["key_hash"] = pd.util.hash_pandas_object(key_columns, index=False).to_numpy(dtype=np.uint64)
            domain_source["request_timestamp"] = pd.to_numeric(domain_source["request_timestamp"], errors="coerce")
            aggregated = domain_source.groupby("key_hash", sort=False).agg(
                source_short=("source_short", "first"),
                traffic_label=("traffic_label", "first"),
                count=("key_hash", "size"),
                min_timestamp=("request_timestamp", "min"),
                max_timestamp=("request_timestamp", "max"),
            )
            for key_hash, row in aggregated.iterrows():
                integer_hash = int(key_hash)
                if integer_hash not in domain_stats:
                    domain_stats[integer_hash] = [
                        str(row.source_short),
                        str(row.traffic_label),
                        int(row["count"]),
                        float(row.min_timestamp) if pd.notna(row.min_timestamp) else math.nan,
                        float(row.max_timestamp) if pd.notna(row.max_timestamp) else math.nan,
                    ]
                else:
                    current = domain_stats[integer_hash]
                    current[2] += int(row["count"])
                    if pd.notna(row.min_timestamp):
                        current[3] = min(current[3], float(row.min_timestamp)) if math.isfinite(current[3]) else float(row.min_timestamp)
                    if pd.notna(row.max_timestamp):
                        current[4] = max(current[4], float(row.max_timestamp)) if math.isfinite(current[4]) else float(row.max_timestamp)

    sample = pd.concat(samples, ignore_index=True)
    for feature in CORE_NUMERIC:
        sample[feature] = pd.to_numeric(sample[feature], errors="coerce")
    hex_chars = set("0123456789abcdefABCDEF")
    base64url_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    sample["hex_char_ratio"] = [character_ratio(value, hex_chars) for value in sample["subdomain"]]
    sample["base64url_char_ratio"] = [character_ratio(value, base64url_chars) for value in sample["subdomain"]]

    domain_rows = []
    for values in domain_stats.values():
        source, label, count, min_timestamp, max_timestamp = values
        duration = max_timestamp - min_timestamp if math.isfinite(min_timestamp) and math.isfinite(max_timestamp) else math.nan
        mean_iat = duration / (count - 1) if count > 1 and duration >= 0 else math.nan
        request_rate = count / duration if duration and duration > 0 else math.nan
        domain_rows.append(
            {
                "source_short": source,
                "traffic_label": label,
                "stratum": stratum(source, label),
                "query_count_per_domain": count,
                "domain_duration": duration,
                "mean_interarrival_time": mean_iat,
                "request_rate": request_rate,
            }
        )
    domains = pd.DataFrame(domain_rows)
    correlation_matrix, correlation_counts = finalize_correlations(pair_stats)

    return {
        "total_rows": total_rows,
        "class_counts": class_counts,
        "source_counts": source_counts,
        "source_groups": source_groups,
        "stratum_counts": stratum_counts,
        "missing_counts": missing_counts,
        "low_card_counts": low_card_counts,
        "category_counts": category_counts,
        "correlation_matrix": correlation_matrix,
        "correlation_counts": correlation_counts,
        "sample": sample,
        "domains": domains,
        "recording_sizes": recording_sizes,
        "recording_strata": recording_strata,
    }


def total_variation(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return math.nan
    return 0.5 * sum(abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total) for key in keys)


def cramers_v(left: Counter, right: Counter) -> float:
    keys = sorted(set(left) | set(right))
    observed = np.array([[left.get(key, 0) for key in keys], [right.get(key, 0) for key in keys]], dtype=float)
    total = observed.sum()
    if total == 0 or observed.shape[1] <= 1:
        return math.nan
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / total
    valid = expected > 0
    chi_square = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    return math.sqrt((chi_square / total) / min(observed.shape[0] - 1, observed.shape[1] - 1))


def choose_sequence_recordings(result: dict, per_stratum: int = 4) -> set[str]:
    candidates: defaultdict[str, list[str]] = defaultdict(list)
    for recording, size in result["recording_sizes"].items():
        if size >= 50:
            candidates[result["recording_strata"][recording]].append(recording)
    selected: set[str] = set()
    for group_name, recordings in candidates.items():
        ordered = sorted(recordings, key=stable_order)
        selected.update(ordered[:per_stratum])
    return selected


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def jaro_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    match_distance = max(len(left), len(right)) // 2 - 1
    match_distance = max(0, match_distance)
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)
    matches = 0
    for left_index, left_char in enumerate(left):
        start = max(0, left_index - match_distance)
        end = min(left_index + match_distance + 1, len(right))
        for right_index in range(start, end):
            if right_matches[right_index] or left_char != right[right_index]:
                continue
            left_matches[left_index] = True
            right_matches[right_index] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    left_matched = [left[index] for index, matched in enumerate(left_matches) if matched]
    right_matched = [right[index] for index, matched in enumerate(right_matches) if matched]
    transpositions = sum(a != b for a, b in zip(left_matched, right_matched)) / 2
    return (matches / len(left) + matches / len(right) + (matches - transpositions) / matches) / 3


def jaro_winkler(left: str, right: str) -> float:
    jaro = jaro_similarity(left, right)
    prefix = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char or prefix == 4:
            break
        prefix += 1
    return jaro + 0.1 * prefix * (1 - jaro)


def lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for right_index, right_char in enumerate(right, start=1):
            current.append(previous[right_index - 1] + 1 if left_char == right_char else max(previous[right_index], current[-1]))
        previous = current
    return previous[-1]


def sequence_metrics(left: str, right: str) -> dict[str, float]:
    left = str(left)[:128]
    right = str(right)[:128]
    maximum_length = max(len(left), len(right), 1)
    longest_substring = SequenceMatcher(None, left, right, autojunk=False).find_longest_match().size
    return {
        "normalized_levenshtein_similarity": 1 - levenshtein_distance(left, right) / maximum_length,
        "jaro_similarity": jaro_similarity(left, right),
        "jaro_winkler_similarity": jaro_winkler(left, right),
        "longest_common_substring_ratio": longest_substring / maximum_length,
        "longest_common_subsequence_ratio": lcs_length(left, right) / maximum_length,
        "reversed_jaro_similarity": jaro_similarity(left[::-1], right[::-1]),
        "reversed_jaro_winkler_similarity": jaro_winkler(left[::-1], right[::-1]),
    }


def second_pass_sequence(result: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    selected_recordings = choose_sequence_recordings(result)
    columns = [
        "source_name",
        "traffic_label",
        "split_group_id",
        "client_ip",
        "base_domain",
        "subdomain",
        "request_timestamp",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(TRAIN_PATH, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False):
        selected = chunk["split_group_id"].astype("string").isin(selected_recordings)
        if selected.any():
            part = chunk.loc[selected].copy()
            part["source_short"] = part["source_name"].map(short_source)
            part["traffic_label"] = part["traffic_label"].astype("string").str.lower().str.strip()
            part["stratum"] = part["source_short"] + " " + part["traffic_label"].str.title()
            chunks.append(part)
    sequence = pd.concat(chunks, ignore_index=True)
    sequence["request_timestamp"] = pd.to_numeric(sequence["request_timestamp"], errors="coerce")
    keys = ["split_group_id", "client_ip", "base_domain"]
    sequence = sequence.dropna(subset=keys + ["request_timestamp"]).sort_values(keys + ["request_timestamp"])
    sequence["interarrival_time"] = sequence.groupby(keys, sort=False)["request_timestamp"].diff()
    sequence["silent_gap"] = (sequence["interarrival_time"] > 60).astype(int)
    sequence["minute_bin"] = np.floor(sequence["request_timestamp"] / 60)
    window_stats = sequence.groupby(keys, sort=False).agg(
        source_short=("source_short", "first"),
        traffic_label=("traffic_label", "first"),
        transaction_count=("request_timestamp", "size"),
        min_timestamp=("request_timestamp", "min"),
        max_timestamp=("request_timestamp", "max"),
        mean_interarrival_time=("interarrival_time", "mean"),
        std_interarrival_time=("interarrival_time", "std"),
        silent_gap_count=("silent_gap", "sum"),
        active_minutes=("minute_bin", "nunique"),
    ).reset_index()
    window_stats["stratum"] = window_stats["source_short"] + " " + window_stats["traffic_label"].str.title()
    duration_minutes = np.floor(window_stats["max_timestamp"] / 60) - np.floor(window_stats["min_timestamp"] / 60) + 1
    window_stats["active_window_ratio"] = window_stats["active_minutes"] / duration_minutes.clip(lower=1)

    sequence["previous_subdomain"] = sequence.groupby(keys, sort=False)["subdomain"].shift(1)
    pairs = sequence.dropna(subset=["previous_subdomain", "subdomain"])
    pairs = pairs.loc[(pairs["subdomain"].astype(str) != "") & (pairs["previous_subdomain"].astype(str) != "")]
    metric_rows: list[dict] = []
    for group_name, group in pairs.groupby("stratum", sort=False):
        sample_size = min(1_000, len(group))
        sampled = group.sample(n=sample_size, random_state=20260819)
        for row in sampled.itertuples(index=False):
            metrics = sequence_metrics(str(row.previous_subdomain), str(row.subdomain))
            metric_rows.append({"stratum": group_name, "source_short": row.source_short, "traffic_label": row.traffic_label, **metrics})
    metric_frame = pd.DataFrame(metric_rows)
    return window_stats, metric_frame, sorted(selected_recordings)


def write_outputs(result: dict, window_stats: pd.DataFrame, sequence_metrics_frame: pd.DataFrame, selected_recordings: list[str]) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = result["total_rows"]
    class_counts = result["class_counts"]

    source_rows = []
    source_order = {
        "Nitsan|||benign": 0,
        "CTU Normal|||benign": 1,
        "Rotem|||malicious": 2,
        "Itamar|||malicious": 3,
        "Nitsan|||malicious": 4,
    }
    for key, count in result["source_counts"].items():
        source, label = key.split("|||", 1)
        source_rows.append(
            {
                "source": source,
                "label": label,
                "transactions": count,
                "recordings": len(result["source_groups"][key]),
                "share_within_class": count / class_counts[label],
            }
        )
    source_rows.sort(key=lambda row: source_order.get(f"{row['source']}|||{row['label']}", 99))
    pd.DataFrame(source_rows).to_csv(OUTPUT_DIR / "table_3_1_source_composition.csv", index=False)

    missing_rows = []
    for group_name, group_total in result["stratum_counts"].items():
        for field in MISSINGNESS_FIELDS:
            missing_rows.append(
                {
                    "stratum": group_name,
                    "field": field,
                    "missing_or_absent_count": result["missing_counts"].get((group_name, field), 0),
                    "missing_or_absent_rate": result["missing_counts"].get((group_name, field), 0) / group_total,
                }
            )
    missing_frame = pd.DataFrame(missing_rows)
    missing_frame.to_csv(OUTPUT_DIR / "missingness_by_source.csv", index=False)

    low_card_rows = []
    for field, counts in result["low_card_counts"].items():
        total = sum(counts.values())
        dominant_value, dominant_count = counts.most_common(1)[0]
        low_card_rows.append(
            {
                "field": field,
                "unique_values": len(counts),
                "dominant_value": dominant_value,
                "dominant_count": dominant_count,
                "dominant_rate": dominant_count / total,
                "top_values": json.dumps(counts.most_common(8), ensure_ascii=False),
            }
        )
    pd.DataFrame(low_card_rows).to_csv(OUTPUT_DIR / "low_cardinality_scan.csv", index=False)

    result["correlation_matrix"].to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    result["correlation_counts"].to_csv(OUTPUT_DIR / "correlation_pair_counts.csv")

    sample = result["sample"].copy()
    transaction_features = [*CORE_NUMERIC, "hex_char_ratio", "base64url_char_ratio"]
    transaction_summary = quantile_summary(sample, "stratum", transaction_features)
    transaction_summary.to_csv(OUTPUT_DIR / "transaction_feature_summary_by_source.csv", index=False)

    effect_rows = []
    comparisons = {
        "Pooled malicious vs benign": (
            sample.loc[sample["traffic_label"] == "malicious"],
            sample.loc[sample["traffic_label"] == "benign"],
        ),
        "Nitsan malicious vs Nitsan benign": (
            sample.loc[(sample["source_short"] == "Nitsan") & (sample["traffic_label"] == "malicious")],
            sample.loc[(sample["source_short"] == "Nitsan") & (sample["traffic_label"] == "benign")],
        ),
    }
    for comparison, (malicious_frame, benign_frame) in comparisons.items():
        for feature in transaction_features:
            malicious_values = pd.to_numeric(malicious_frame[feature], errors="coerce").to_numpy(dtype=float)
            benign_values = pd.to_numeric(benign_frame[feature], errors="coerce").to_numpy(dtype=float)
            delta = cliffs_delta(malicious_values, benign_values)
            u_statistic, p_value = mann_whitney_normal_approx(malicious_values, benign_values)
            effect_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "malicious_median": float(np.nanmedian(malicious_values)) if np.isfinite(malicious_values).any() else math.nan,
                    "benign_median": float(np.nanmedian(benign_values)) if np.isfinite(benign_values).any() else math.nan,
                    "cliffs_delta": delta,
                    "effect_magnitude": effect_magnitude(delta),
                    "mann_whitney_u": u_statistic,
                    "p_value_normal_approx": p_value,
                }
            )
    effect_frame = pd.DataFrame(effect_rows)
    effect_frame.to_csv(OUTPUT_DIR / "transaction_feature_effect_sizes.csv", index=False)

    domain_features = ["query_count_per_domain", "domain_duration", "mean_interarrival_time", "request_rate"]
    domain_summary = quantile_summary(result["domains"], "stratum", domain_features)
    domain_summary.to_csv(OUTPUT_DIR / "domain_recurrence_timing_summary.csv", index=False)
    domain_effect_rows = []
    for comparison, malicious_filter, benign_filter in [
        ("Pooled malicious vs benign", result["domains"]["traffic_label"] == "malicious", result["domains"]["traffic_label"] == "benign"),
        (
            "Nitsan malicious vs Nitsan benign",
            (result["domains"]["source_short"] == "Nitsan") & (result["domains"]["traffic_label"] == "malicious"),
            (result["domains"]["source_short"] == "Nitsan") & (result["domains"]["traffic_label"] == "benign"),
        ),
    ]:
        for feature in domain_features:
            malicious_values = pd.to_numeric(result["domains"].loc[malicious_filter, feature], errors="coerce").to_numpy(dtype=float)
            benign_values = pd.to_numeric(result["domains"].loc[benign_filter, feature], errors="coerce").to_numpy(dtype=float)
            delta = cliffs_delta(malicious_values, benign_values, cap=30_000)
            _, p_value = mann_whitney_normal_approx(malicious_values, benign_values, cap=30_000)
            domain_effect_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "malicious_median": float(np.nanmedian(malicious_values)) if np.isfinite(malicious_values).any() else math.nan,
                    "benign_median": float(np.nanmedian(benign_values)) if np.isfinite(benign_values).any() else math.nan,
                    "cliffs_delta": delta,
                    "effect_magnitude": effect_magnitude(delta),
                    "p_value_normal_approx": p_value,
                }
            )
    pd.DataFrame(domain_effect_rows).to_csv(OUTPUT_DIR / "domain_feature_effect_sizes.csv", index=False)

    categorical_rows = []
    for field in ("query_type", "dns_rcode"):
        for scope in ("pooled", "nitsan"):
            benign_counts = result["category_counts"][f"{field}_{scope}_benign"]
            malicious_counts = result["category_counts"][f"{field}_{scope}_malicious"]
            categorical_rows.append(
                {
                    "field": field,
                    "scope": scope,
                    "total_variation_distance": total_variation(malicious_counts, benign_counts),
                    "cramers_v": cramers_v(malicious_counts, benign_counts),
                    "benign_top": json.dumps(benign_counts.most_common(8), ensure_ascii=False),
                    "malicious_top": json.dumps(malicious_counts.most_common(8), ensure_ascii=False),
                }
            )
    pd.DataFrame(categorical_rows).to_csv(OUTPUT_DIR / "categorical_distribution_effects.csv", index=False)

    window_features = [
        "transaction_count",
        "mean_interarrival_time",
        "std_interarrival_time",
        "silent_gap_count",
        "active_window_ratio",
    ]
    window_summary = quantile_summary(window_stats, "stratum", window_features)
    window_summary.to_csv(OUTPUT_DIR / "sequence_window_summary.csv", index=False)
    sequence_feature_names = [column for column in sequence_metrics_frame.columns if column not in {"stratum", "source_short", "traffic_label"}]
    sequence_summary = quantile_summary(sequence_metrics_frame, "stratum", sequence_feature_names)
    sequence_summary.to_csv(OUTPUT_DIR / "sequence_similarity_summary.csv", index=False)

    sequence_effect_rows = []
    for comparison, malicious_filter, benign_filter in [
        (
            "Pooled malicious vs benign",
            sequence_metrics_frame["traffic_label"] == "malicious",
            sequence_metrics_frame["traffic_label"] == "benign",
        ),
        (
            "Nitsan malicious vs Nitsan benign",
            (sequence_metrics_frame["source_short"] == "Nitsan") & (sequence_metrics_frame["traffic_label"] == "malicious"),
            (sequence_metrics_frame["source_short"] == "Nitsan") & (sequence_metrics_frame["traffic_label"] == "benign"),
        ),
    ]:
        for feature in sequence_feature_names:
            malicious_values = pd.to_numeric(sequence_metrics_frame.loc[malicious_filter, feature], errors="coerce").to_numpy(dtype=float)
            benign_values = pd.to_numeric(sequence_metrics_frame.loc[benign_filter, feature], errors="coerce").to_numpy(dtype=float)
            delta = cliffs_delta(malicious_values, benign_values, cap=5_000)
            _, p_value = mann_whitney_normal_approx(malicious_values, benign_values, cap=5_000)
            sequence_effect_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "malicious_median": float(np.nanmedian(malicious_values)) if np.isfinite(malicious_values).any() else math.nan,
                    "benign_median": float(np.nanmedian(benign_values)) if np.isfinite(benign_values).any() else math.nan,
                    "cliffs_delta": delta,
                    "effect_magnitude": effect_magnitude(delta),
                    "p_value_normal_approx": p_value,
                }
            )
    pd.DataFrame(sequence_effect_rows).to_csv(OUTPUT_DIR / "sequence_similarity_effect_sizes.csv", index=False)
    (OUTPUT_DIR / "sequence_recordings_used.txt").write_text("\n".join(selected_recordings), encoding="utf-8")

    summary = {
        "total_rows": total_rows,
        "class_counts": dict(class_counts),
        "unique_recordings": len(result["recording_sizes"]),
        "transaction_sample_rows": len(sample),
        "domain_groups": len(result["domains"]),
        "sequence_recordings": len(selected_recordings),
        "sequence_pairs": len(sequence_metrics_frame),
        "source_composition": source_rows,
        "label_correlations": {
            feature: float(result["correlation_matrix"].loc["malicious_label", feature])
            for feature in CORRELATION_FEATURES
            if feature != "malicious_label"
        },
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = first_pass()
    window_stats, sequence_metrics_frame, selected_recordings = second_pass_sequence(result)
    summary = write_outputs(result, window_stats, sequence_metrics_frame, selected_recordings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
