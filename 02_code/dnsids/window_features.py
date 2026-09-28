"""Window statistics shared by the Milestone 3 input representations.

The numeric definitions are preserved from the accepted feature pipeline.
This module performs no file I/O and has no machine-specific paths.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

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


META_COLUMNS = ["split_group_id", "traffic_label", "source_name", "window_number", "window_tx_count"]


def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def stat(values: np.ndarray, operation: str) -> float:
    values = finite(values)
    if values.size == 0:
        return math.nan
    if operation == "mean":
        return float(np.mean(values))
    if operation == "std":
        return float(np.std(values))
    if operation == "median":
        return float(np.median(values))
    if operation == "p95":
        return float(np.quantile(values, 0.95))
    raise ValueError(operation)


def ratio(values: pd.Series, value: str) -> float:
    normalized = values.fillna("NULL").astype(str).str.upper()
    return float((normalized == value).mean())


def adjacent_jaro_winkler(subdomains: pd.Series) -> float:
    values = subdomains.fillna("").astype(str).tolist()
    if len(values) < 2:
        return math.nan
    pair_indices = np.arange(len(values) - 1)
    if len(pair_indices) > 24:
        pair_indices = np.unique(np.linspace(0, len(pair_indices) - 1, 24).round().astype(int))
    scores = [jaro_winkler(values[index][:128], values[index + 1][:128]) for index in pair_indices]
    return float(np.mean(scores)) if scores else math.nan


def aggregate_window(window: pd.DataFrame, window_number: int) -> dict[str, float | str | int]:
    qname = safe_numeric(window["query_name_len"])
    subdomain = safe_numeric(window["subdomain_len"])
    labels = safe_numeric(window["num_labels"])
    max_label = safe_numeric(window["max_label_len"])
    query_entropy = safe_numeric(window["query_entropy"])
    longest_entropy = safe_numeric(window["longest_subdomain_entropy"])
    consonants = safe_numeric(window["long_consonant_string_count"])
    response_time = safe_numeric(window["req_resp_time_diff"])
    bytes_total = safe_numeric(window["total_transaction_bytes"])
    request_payload = safe_numeric(window["request_payload_len"])
    response_payload = safe_numeric(window["response_payload_len"])
    timestamps = safe_numeric(window["request_timestamp"])
    timestamps = np.sort(finite(timestamps))
    iat = np.diff(timestamps)
    iat = iat[(iat >= 0) & np.isfinite(iat)]

    base_domains = window["base_domain"].fillna("").astype(str)
    subdomains = window["subdomain"].fillna("").astype(str)
    query_names = window["query_name_normalized"].fillna("").astype(str)
    qtypes = window["query_type"].fillna("NULL").astype(str).str.upper()
    rcodes = window["dns_rcode"].fillna("NULL").astype(str).str.upper()
    count = len(window)
    base_counts = base_domains.value_counts(dropna=False)

    result: dict[str, float | str | int] = {
        "split_group_id": str(window["split_group_id"].iloc[0]),
        "traffic_label": str(window["traffic_label"].iloc[0]).lower(),
        "source_name": str(window["source_name"].iloc[0]),
        "window_number": int(window_number),
        "window_tx_count": int(count),
        "query_name_len_mean": stat(qname, "mean"),
        "query_name_len_std": stat(qname, "std"),
        "query_name_len_p95": stat(qname, "p95"),
        "subdomain_len_mean": stat(subdomain, "mean"),
        "subdomain_len_std": stat(subdomain, "std"),
        "subdomain_len_p95": stat(subdomain, "p95"),
        "num_labels_mean": stat(labels, "mean"),
        "num_labels_std": stat(labels, "std"),
        "num_labels_p95": stat(labels, "p95"),
        "max_label_len_mean": stat(max_label, "mean"),
        "max_label_len_p95": stat(max_label, "p95"),
        "query_entropy_mean": stat(query_entropy, "mean"),
        "query_entropy_std": stat(query_entropy, "std"),
        "query_entropy_p95": stat(query_entropy, "p95"),
        "high_query_entropy_ratio": float(np.nanmean(query_entropy >= 4.0)),
        "longest_subdomain_entropy_mean": stat(longest_entropy, "mean"),
        "longest_subdomain_entropy_p95": stat(longest_entropy, "p95"),
        "high_subdomain_entropy_ratio": float(np.nanmean(longest_entropy >= 3.5)),
        "long_consonant_count_mean": stat(consonants, "mean"),
        "long_consonant_positive_ratio": float(np.nanmean(consonants > 0)),
        "transaction_bytes_mean": stat(bytes_total, "mean"),
        "transaction_bytes_std": stat(bytes_total, "std"),
        "transaction_bytes_p95": stat(bytes_total, "p95"),
        "request_payload_mean": stat(request_payload, "mean"),
        "response_payload_mean": stat(response_payload, "mean"),
        "response_missing_ratio": float(pd.to_numeric(window["is_response_missing"], errors="coerce").fillna(1).mean()),
        "response_time_median": stat(response_time, "median"),
        "response_time_std": stat(response_time, "std"),
        "response_time_p95": stat(response_time, "p95"),
        "unique_base_domain_ratio": float(base_domains.nunique(dropna=False) / count),
        "top_base_domain_share": float(base_counts.iloc[0] / count) if not base_counts.empty else math.nan,
        "unique_subdomain_ratio": float(subdomains.nunique(dropna=False) / count),
        "repeated_query_ratio": float(1 - query_names.nunique(dropna=False) / count),
        "iat_mean": stat(iat, "mean"),
        "iat_std": stat(iat, "std"),
        "iat_median": stat(iat, "median"),
        "iat_p95": stat(iat, "p95"),
        "iat_cv": float(np.std(iat) / np.mean(iat)) if iat.size and np.mean(iat) > 0 else math.nan,
        "adjacent_subdomain_jaro_winkler_mean": adjacent_jaro_winkler(subdomains),
    }

    for value in ["A", "AAAA", "CNAME", "TXT", "MX", "NULL", "PRIVATE"]:
        result[f"qtype_{value.lower()}_ratio"] = ratio(qtypes, value)
    known_qtypes = {"A", "AAAA", "CNAME", "TXT", "MX", "NULL", "PRIVATE"}
    result["qtype_other_ratio"] = float((~qtypes.isin(known_qtypes)).mean())
    for value in ["NOERROR", "NXDOMAIN", "SERVFAIL", "NULL"]:
        result[f"rcode_{value.lower()}_ratio"] = ratio(rcodes, value)
    known_rcodes = {"NOERROR", "NXDOMAIN", "SERVFAIL", "NULL"}
    result["rcode_other_ratio"] = float((~rcodes.isin(known_rcodes)).mean())
    return result
