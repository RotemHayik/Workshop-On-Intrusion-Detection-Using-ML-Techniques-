from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from generate_chapter3_full_analysis import jaro_winkler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = Path(r"G:\My Drive\Workshop Files\Rotem\train_transactions.csv")
ORIGINAL_OUTPUT_DIR = ROOT / "chapter4_outputs"
OUTPUT_DIR = ROOT / "comprehensive_corrections" / "chapter4_group_safe"
GROUP_EVIDENCE_PATH = ROOT / "comprehensive_corrections" / "group_aware_evidence" / "Appendix_C_Group_Aware_52_Feature_Evidence.csv"
SEED = 1_071_004
WINDOW_SIZE = 128
MIN_WINDOW_SIZE = 32
MAX_WINDOWS_PER_GROUP = 24
CHUNK_SIZE = 120_000

USECOLS = [
    "split_group_id", "traffic_label", "source_name", "request_timestamp",
    "base_domain", "subdomain", "query_name_normalized", "query_type", "dns_rcode",
    "query_name_len", "subdomain_len", "num_labels", "max_label_len",
    "query_entropy", "longest_subdomain_entropy", "long_consonant_string_count",
    "req_resp_time_diff", "total_transaction_bytes", "request_payload_len",
    "response_payload_len", "is_response_missing",
]

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


def process_group(group: pd.DataFrame) -> list[dict[str, float | str | int]]:
    if group["traffic_label"].astype(str).str.lower().nunique() != 1:
        raise ValueError(f"Mixed labels inside {group['split_group_id'].iloc[0]}")
    group = group.sort_values("request_timestamp", kind="mergesort").reset_index(drop=True)
    windows: list[tuple[int, pd.DataFrame]] = []
    for start in range(0, len(group), WINDOW_SIZE):
        window = group.iloc[start : start + WINDOW_SIZE]
        if len(window) >= MIN_WINDOW_SIZE:
            windows.append((start // WINDOW_SIZE, window))
    if not windows:
        return []
    selected = np.arange(len(windows))
    if len(selected) > MAX_WINDOWS_PER_GROUP:
        selected = np.unique(np.linspace(0, len(selected) - 1, MAX_WINDOWS_PER_GROUP).round().astype(int))
    return [aggregate_window(windows[index][1], windows[index][0]) for index in selected]


def build_window_dataset() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    bucket_dir = OUTPUT_DIR / "group_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    bucket_paths = [bucket_dir / f"bucket_{index:02d}.csv" for index in range(16)]
    for bucket_path in bucket_paths:
        if bucket_path.exists():
            bucket_path.unlink()
    read_options = dict(usecols=USECOLS, chunksize=CHUNK_SIZE, low_memory=False)
    for chunk_number, chunk in enumerate(pd.read_csv(TRAIN_PATH, **read_options), start=1):
        group_strings = chunk["split_group_id"].astype(str)
        unique_groups = group_strings.unique()
        group_bucket = {group: zlib.crc32(group.encode("utf-8")) % len(bucket_paths) for group in unique_groups}
        bucket_numbers = group_strings.map(group_bucket).to_numpy(dtype=int)
        for bucket_number in np.unique(bucket_numbers):
            bucket_path = bucket_paths[int(bucket_number)]
            subset = chunk.loc[bucket_numbers == bucket_number]
            subset.to_csv(bucket_path, mode="a", header=not bucket_path.exists(), index=False, encoding="utf-8")
        if chunk_number % 3 == 0:
            print(f"partitioned chunks={chunk_number}", flush=True)
    processed_groups = 0
    for bucket_number, bucket_path in enumerate(bucket_paths):
        if not bucket_path.exists():
            continue
        bucket = pd.read_csv(bucket_path, low_memory=False)
        for _, group in bucket.groupby("split_group_id", sort=False, dropna=False):
            rows.extend(process_group(group))
            processed_groups += 1
        bucket_path.unlink()
        print(f"processed bucket={bucket_number + 1}/{len(bucket_paths)}, groups={processed_groups}, windows={len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "train_window_features.csv", index=False, encoding="utf-8-sig")
    return frame


def weighted_gini(y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    positive = float(weights[y == 1].sum()) / total
    return 1.0 - positive * positive - (1.0 - positive) * (1.0 - positive)


@dataclass
class TreeNode:
    probability: float
    feature: int = -1
    threshold: float = math.nan
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class RandomizedTree:
    def __init__(self, max_depth: int, min_leaf: int, max_features: int, rng: np.random.Generator):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.rng = rng
        self.importance: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> "RandomizedTree":
        self.importance = np.zeros(x.shape[1], dtype=float)
        self.root = self._build(x, y, weights, 0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, depth: int) -> TreeNode:
        total_weight = float(weights.sum())
        probability = float(weights[y == 1].sum() / total_weight) if total_weight > 0 else float(np.mean(y))
        node = TreeNode(probability=probability)
        parent_gini = weighted_gini(y, weights)
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or parent_gini < 1e-9:
            return node
        feature_indices = self.rng.choice(x.shape[1], size=min(self.max_features, x.shape[1]), replace=False)
        best: tuple[float, int, float, np.ndarray] | None = None
        for feature in feature_indices:
            values = x[:, feature]
            unique = np.unique(values)
            if unique.size < 2:
                continue
            quantiles = np.linspace(0.05, 0.95, min(17, unique.size - 1))
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(left_mask.sum())
                right_count = len(y) - left_count
                if left_count < self.min_leaf or right_count < self.min_leaf:
                    continue
                left_weight = float(weights[left_mask].sum())
                right_weight = total_weight - left_weight
                if left_weight <= 0 or right_weight <= 0:
                    continue
                child_gini = (
                    left_weight * weighted_gini(y[left_mask], weights[left_mask])
                    + right_weight * weighted_gini(y[~left_mask], weights[~left_mask])
                ) / total_weight
                gain = parent_gini - child_gini
                if best is None or gain > best[0]:
                    best = (gain, int(feature), float(threshold), left_mask)
        if best is None or best[0] <= 1e-7:
            return node
        gain, feature, threshold, left_mask = best
        assert self.importance is not None
        self.importance[feature] += total_weight * gain
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(x[left_mask], y[left_mask], weights[left_mask], depth + 1)
        node.right = self._build(x[~left_mask], y[~left_mask], weights[~left_mask], depth + 1)
        return node

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result = np.empty(len(x), dtype=float)
        for row_index, row in enumerate(x):
            node = self.root
            while node.feature >= 0:
                node = node.left if row[node.feature] <= node.threshold else node.right
                assert node is not None
            result[row_index] = node.probability
        return result


class GroupAwareRandomForest:
    def __init__(self, n_trees: int, max_depth: int = 10, min_leaf: int = 8, seed: int = SEED):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> "GroupAwareRandomForest":
        rng = np.random.default_rng(self.seed)
        self.trees: list[RandomizedTree] = []
        self.importances_ = np.zeros(x.shape[1], dtype=float)
        max_features = max(1, int(round(math.sqrt(x.shape[1]))))
        for tree_number in range(self.n_trees):
            indices = rng.choice(len(y), size=len(y), replace=True)
            tree = RandomizedTree(self.max_depth, self.min_leaf, max_features, np.random.default_rng(rng.integers(1, 2**31)))
            tree.fit(x[indices], y[indices], weights[indices])
            self.trees.append(tree)
            assert tree.importance is not None
            self.importances_ += tree.importance
            if (tree_number + 1) % 100 == 0:
                print(f"fitted trees={tree_number + 1}/{self.n_trees}", flush=True)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ /= total
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict_proba(x) for tree in self.trees], axis=0)


def make_group_weights(frame: pd.DataFrame) -> np.ndarray:
    group_sizes = frame.groupby("split_group_id")["split_group_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / group_sizes
    labels = (frame["traffic_label"].astype(str).str.lower() == "malicious").to_numpy(dtype=int)
    for label in [0, 1]:
        total = weights[labels == label].sum()
        if total > 0:
            weights[labels == label] *= 1.0 / total
    return weights / np.mean(weights)


def impute_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    matrix = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    medians: dict[str, float] = {}
    for column_index, column in enumerate(feature_columns):
        values = matrix[:, column_index]
        valid = values[np.isfinite(values)]
        median = float(np.median(valid)) if valid.size else 0.0
        medians[column] = median
        values[~np.isfinite(values)] = median
        matrix[:, column_index] = values
    return matrix, medians


def apply_imputation(frame: pd.DataFrame, feature_columns: list[str], medians: dict[str, float]) -> np.ndarray:
    matrix = frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    for column_index, column in enumerate(feature_columns):
        values = matrix[:, column_index]
        values[~np.isfinite(values)] = float(medians[column])
        matrix[:, column_index] = values
    return matrix


def balanced_accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = probabilities >= 0.5
    sensitivity = float(np.mean(predictions[y == 1] == 1)) if np.any(y == 1) else math.nan
    specificity = float(np.mean(predictions[y == 0] == 0)) if np.any(y == 0) else math.nan
    return float(np.nanmean([sensitivity, specificity]))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    losses = -(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
    class_losses = [float(losses[y == label].mean()) for label in [0, 1] if np.any(y == label)]
    return float(np.mean(class_losses))


def stratified_group_folds(frame: pd.DataFrame, n_splits: int = 5) -> list[set[str]]:
    groups = frame[["split_group_id", "traffic_label"]].drop_duplicates()
    rng = np.random.default_rng(SEED)
    folds: list[set[str]] = [set() for _ in range(n_splits)]
    for label in ["benign", "malicious"]:
        values = groups.loc[groups["traffic_label"].astype(str).str.lower() == label, "split_group_id"].astype(str).tolist()
        rng.shuffle(values)
        for index, group in enumerate(values):
            folds[index % n_splits].add(group)
    return folds


def model_analysis(frame: pd.DataFrame) -> dict:
    evidence = pd.read_csv(GROUP_EVIDENCE_PATH, low_memory=False)
    core_features = evidence.loc[
        evidence["final_empirical_decision"] != "Do not claim as independently justified; supporting/ablation only",
        "feature",
    ].astype(str).tolist()
    candidate_features = [column for column in core_features if column in frame.columns and column not in META_COLUMNS]
    if len(candidate_features) != 42:
        raise RuntimeError(f"Expected 42 empirically supported core features, found {len(candidate_features)}")
    numeric = frame[candidate_features].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in candidate_features if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1]
    x, medians = impute_matrix(frame, usable)
    y = (frame["traffic_label"].astype(str).str.lower() == "malicious").to_numpy(dtype=int)
    groups = frame["split_group_id"].astype(str).to_numpy()
    weights = make_group_weights(frame)

    folds = stratified_group_folds(frame, 5)
    fold_rows: list[dict] = []
    fold_importances: list[np.ndarray] = []
    permutation_by_fold: list[np.ndarray] = []
    for fold_index, test_groups in enumerate(folds):
        test_mask = np.isin(groups, list(test_groups))
        train_mask = ~test_mask
        train_frame = frame.loc[train_mask].reset_index(drop=True)
        held_frame = frame.loc[test_mask].reset_index(drop=True)
        x_train, fold_medians = impute_matrix(train_frame, usable)
        x_held = apply_imputation(held_frame, usable, fold_medians)
        y_train = (train_frame["traffic_label"].astype(str).str.lower() == "malicious").to_numpy(dtype=int)
        y_held = (held_frame["traffic_label"].astype(str).str.lower() == "malicious").to_numpy(dtype=int)
        fold_weights = make_group_weights(train_frame)
        model = GroupAwareRandomForest(n_trees=100, seed=SEED + fold_index)
        model.fit(x_train, y_train, fold_weights)
        probabilities = model.predict_proba(x_held)
        fold_rows.append({
            "fold": fold_index + 1,
            "train_windows": int(train_mask.sum()),
            "held_windows": int(test_mask.sum()),
            "train_groups": int(len(set(groups[train_mask]))),
            "held_groups": int(len(set(groups[test_mask]))),
            "group_overlap": int(len(set(groups[train_mask]) & set(groups[test_mask]))),
            "balanced_accuracy": balanced_accuracy(y_held, probabilities),
            "roc_auc": roc_auc(y_held, probabilities),
        })
        fold_importances.append(model.importances_)
        baseline = balanced_log_loss(y_held, probabilities)
        rng = np.random.default_rng(SEED + 444 + fold_index)
        fold_permutation = np.zeros((len(usable), 5), dtype=float)
        permuted_x = x_held.copy()
        for feature_index in range(len(usable)):
            original = permuted_x[:, feature_index].copy()
            for repeat in range(5):
                permuted_x[:, feature_index] = rng.permutation(original)
                fold_permutation[feature_index, repeat] = balanced_log_loss(y_held, model.predict_proba(permuted_x)) - baseline
            permuted_x[:, feature_index] = original
        permutation_by_fold.append(fold_permutation)

    final_model = GroupAwareRandomForest(n_trees=300, seed=SEED + 99)
    final_model.fit(x, y, weights)

    permutation = np.concatenate(permutation_by_fold, axis=1)

    nitsan_mask = frame["source_name"].astype(str).str.lower().eq("nitsan").to_numpy()
    nitsan_weights = make_group_weights(frame.loc[nitsan_mask].copy())
    nitsan_model = GroupAwareRandomForest(n_trees=250, seed=SEED + 555)
    nitsan_model.fit(x[nitsan_mask], y[nitsan_mask], nitsan_weights)

    importance = pd.DataFrame({
        "feature": usable,
        "mdi_importance": final_model.importances_,
        "mdi_fold_mean": np.mean(fold_importances, axis=0),
        "mdi_fold_std": np.std(fold_importances, axis=0),
        "held_group_permutation_mean": permutation.mean(axis=1),
        "held_group_permutation_std": permutation.std(axis=1),
        "nitsan_mdi_importance": nitsan_model.importances_,
    })
    importance["mdi_rank"] = importance["mdi_importance"].rank(method="min", ascending=False).astype(int)
    importance["permutation_rank"] = importance["held_group_permutation_mean"].rank(method="min", ascending=False).astype(int)
    importance["nitsan_rank"] = importance["nitsan_mdi_importance"].rank(method="min", ascending=False).astype(int)
    importance["nitsan_rank_shift"] = importance["nitsan_rank"] - importance["mdi_rank"]
    importance = importance.sort_values("mdi_importance", ascending=False).reset_index(drop=True)
    importance.to_csv(OUTPUT_DIR / "feature_importance_diagnostics.csv", index=False, encoding="utf-8-sig")

    correlations = pd.DataFrame(x, columns=usable).corr().abs()
    pairs = []
    for left_index, left in enumerate(usable):
        for right_index in range(left_index + 1, len(usable)):
            right = usable[right_index]
            value = float(correlations.iloc[left_index, right_index])
            if value >= 0.90:
                pairs.append({"feature_a": left, "feature_b": right, "absolute_correlation": value})
    correlation_pairs = pd.DataFrame(pairs).sort_values("absolute_correlation", ascending=False) if pairs else pd.DataFrame(columns=["feature_a", "feature_b", "absolute_correlation"])
    correlation_pairs.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig")

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(OUTPUT_DIR / "group_cv_metrics.csv", index=False, encoding="utf-8-sig")
    medians_frame = pd.DataFrame({"feature": list(medians), "train_median_imputation": list(medians.values())})
    medians_frame.to_csv(OUTPUT_DIR / "train_imputation_values.csv", index=False, encoding="utf-8-sig")

    rank_matrix = pd.DataFrame(np.vstack(fold_importances).T).rank(method="average", ascending=False)
    stability = rank_matrix.corr(method="spearman")
    stability_values = stability.to_numpy()[np.triu_indices_from(stability, k=1)]

    summary = {
        "train_transactions": 1_401_186,
        "window_size": WINDOW_SIZE,
        "minimum_partial_window": MIN_WINDOW_SIZE,
        "max_windows_per_group": MAX_WINDOWS_PER_GROUP,
        "window_rows": int(len(frame)),
        "groups": int(frame["split_group_id"].nunique()),
        "benign_windows": int((y == 0).sum()),
        "malicious_windows": int((y == 1).sum()),
        "feature_count": len(usable),
        "candidate_core_feature_count": len(candidate_features),
        "preprocessing_fitted_inside_each_fold": True,
        "permutation_validation_folds": len(permutation_by_fold),
        "permutation_repeats_per_fold": 5,
        "forest_implementation": "Custom bootstrap Random-Forest-style ensemble with sqrt feature subsampling and quantile candidate thresholds",
        "forest_max_depth": 10,
        "forest_min_leaf": 8,
        "cv_balanced_accuracy_mean": float(fold_frame["balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(fold_frame["balanced_accuracy"].std(ddof=0)),
        "cv_roc_auc_mean": float(fold_frame["roc_auc"].mean()),
        "cv_roc_auc_std": float(fold_frame["roc_auc"].std(ddof=0)),
        "median_fold_rank_spearman": float(np.median(stability_values)),
        "max_group_overlap": int(fold_frame["group_overlap"].max()),
        "nitsan_windows": int(nitsan_mask.sum()),
        "nitsan_groups": int(frame.loc[nitsan_mask, "split_group_id"].nunique()),
        "high_correlation_pair_count": int(len(correlation_pairs)),
        "top_features": importance.head(15).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "chapter4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary": summary, "importance": importance, "correlations": correlation_pairs}


DISPLAY_NAMES = {
    "longest_subdomain_entropy_mean": "Mean longest-subdomain entropy",
    "longest_subdomain_entropy_p95": "95th percentile subdomain entropy",
    "query_entropy_mean": "Mean query entropy",
    "query_entropy_p95": "95th percentile query entropy",
    "query_name_len_mean": "Mean query-name length",
    "query_name_len_p95": "95th percentile query-name length",
    "subdomain_len_mean": "Mean subdomain length",
    "subdomain_len_p95": "95th percentile subdomain length",
    "num_labels_mean": "Mean number of DNS labels",
    "num_labels_p95": "95th percentile number of labels",
    "transaction_bytes_mean": "Mean transaction bytes",
    "transaction_bytes_p95": "95th percentile transaction bytes",
    "top_base_domain_share": "Top base-domain share",
    "unique_base_domain_ratio": "Unique base-domain ratio",
    "unique_subdomain_ratio": "Unique subdomain ratio",
    "repeated_query_ratio": "Repeated-query ratio",
    "adjacent_subdomain_jaro_winkler_mean": "Adjacent-subdomain Jaro-Winkler",
    "qtype_cname_ratio": "CNAME query ratio",
    "qtype_txt_ratio": "TXT query ratio",
    "qtype_a_ratio": "A query ratio",
    "response_time_median": "Median response time",
    "response_time_p95": "95th percentile response time",
    "iat_mean": "Mean inter-arrival time",
    "iat_median": "Median inter-arrival time",
    "iat_cv": "Inter-arrival coefficient of variation",
    "response_missing_ratio": "Missing-response ratio",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path(r"C:\Windows\Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def create_importance_figure(importance: pd.DataFrame) -> None:
    top = importance.head(15).copy().iloc[::-1]
    positive_perm = importance["held_group_permutation_mean"].clip(lower=0)
    perm_total = float(positive_perm.sum()) or 1.0
    top["permutation_share"] = top["held_group_permutation_mean"].clip(lower=0) / perm_total
    width, height = 1900, 1250
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    navy, blue, gold, gray, dark = "#17324D", "#39749B", "#E29B33", "#D8E0E7", "#1D2730"
    draw.text((90, 50), "Random Forest Feature Ranking on Train-Only Recording-Safe Windows", font=font(42, True), fill=navy)
    draw.text((90, 110), "Impurity importance (bars) and held-recording permutation share (dots)", font=font(27), fill="#52606D")
    plot_left, plot_right, plot_top, plot_bottom = 760, 1810, 205, 1120
    max_value = max(float(top["mdi_importance"].max()), float(top["permutation_share"].max())) * 1.12
    max_value = max(max_value, 0.01)
    for tick in np.linspace(0, max_value, 5):
        x = plot_left + (plot_right - plot_left) * tick / max_value
        draw.line((x, plot_top, x, plot_bottom), fill=gray, width=2)
        draw.text((x - 24, plot_bottom + 14), f"{tick * 100:.1f}%", font=font(22), fill="#64707C")
    row_height = (plot_bottom - plot_top) / len(top)
    for position, row in enumerate(top.itertuples(index=False)):
        y = plot_top + position * row_height + row_height / 2
        label = DISPLAY_NAMES.get(row.feature, row.feature.replace("_", " ").title())
        label_box = draw.textbbox((0, 0), label, font=font(24))
        draw.text((plot_left - 28 - (label_box[2] - label_box[0]), y - 15), label, font=font(24), fill=dark)
        bar_end = plot_left + (plot_right - plot_left) * float(row.mdi_importance) / max_value
        draw.rounded_rectangle((plot_left, y - 14, bar_end, y + 14), radius=7, fill=blue)
        dot_x = plot_left + (plot_right - plot_left) * float(row.permutation_share) / max_value
        draw.ellipse((dot_x - 9, y - 9, dot_x + 9, y + 9), fill=gold, outline="white", width=2)
    draw.rectangle((95, 1180, 125, 1200), fill=blue)
    draw.text((140, 1171), "Mean decrease in impurity", font=font(22), fill=dark)
    draw.ellipse((510, 1180, 530, 1200), fill=gold)
    draw.text((545, 1171), "Held-recording permutation importance", font=font(22), fill=dark)
    image.save(OUTPUT_DIR / "Figure_4_1_RF_Feature_Importance.png", dpi=(220, 220))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cached_windows = OUTPUT_DIR / "train_window_features.csv"
    original_cache = ORIGINAL_OUTPUT_DIR / "train_window_features.csv"
    if cached_windows.exists():
        frame = pd.read_csv(cached_windows, low_memory=False)
    elif original_cache.exists():
        frame = pd.read_csv(original_cache, low_memory=False)
        frame.to_csv(cached_windows, index=False, encoding="utf-8-sig")
    else:
        frame = build_window_dataset()
    result = model_analysis(frame)
    create_importance_figure(result["importance"])
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
