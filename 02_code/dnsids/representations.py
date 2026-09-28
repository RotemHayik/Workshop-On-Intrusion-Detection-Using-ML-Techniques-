"""Portable tabular, sequence and graph input construction for Milestone 3.

Feature definitions, window ordering and numeric transformations are preserved
from the accepted preparation pipeline. Every output path is supplied by the
caller; fitting is performed on the Train inputs passed to the fit functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .window_features import META_COLUMNS, aggregate_window

WINDOW_SIZE = 128


MIN_WINDOW_SIZE = 32


CHUNK_SIZE = 50_000


TABULAR_INPUT_COLUMNS = [
    "split_group_id", "traffic_label", "source_name", "tool_label", "scenario_type",
    "evaluation_subset", "request_timestamp", "base_domain", "subdomain",
    "query_name_normalized", "query_type", "dns_rcode", "query_name_len",
    "subdomain_len", "num_labels", "max_label_len", "query_entropy",
    "longest_subdomain_entropy", "long_consonant_string_count",
    "req_resp_time_diff", "total_transaction_bytes", "request_payload_len",
    "response_payload_len", "is_response_missing", "client_ip", "server_ip",
    "request_len", "response_len",
]


SEQUENCE_CONTINUOUS = [
    "query_name_len", "subdomain_len", "num_labels", "max_label_len",
    "query_entropy", "longest_subdomain_entropy", "long_consonant_string_count",
    "request_len", "request_payload_len", "response_len", "response_payload_len",
    "total_transaction_bytes", "req_resp_time_diff", "interarrival_time",
    "is_response_missing",
]


ENTROPY_FEATURES = {"query_entropy", "longest_subdomain_entropy"}


BINARY_FEATURES = {"is_response_missing"}


QTYPE_VOCAB = {
    "PAD": 0, "UNK": 1, "A": 2, "AAAA": 3, "CNAME": 4, "TXT": 5,
    "MX": 6, "NULL": 7, "PRIVATE": 8, "NS": 9, "SOA": 10, "PTR": 11,
    "KEY": 12, "SRV": 13, "SVCB": 14, "HTTPS": 15, "ANY": 16,
}


RCODE_VOCAB = {
    "PAD": 0, "UNK": 1, "NOERROR": 2, "NXDOMAIN": 3, "SERVFAIL": 4,
    "REFUSED": 5, "FORMERR": 6, "NOTIMP": 7, "NULL": 8,
}


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def stable_sample_id(split: str, group_id: str, window_number: int) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16]
    return f"{split}-{digest}-{window_number:06d}"


def iter_groups(path: Path, usecols: list[str], tag: str, *, output: Path):
    # Keep scratch paths short enough for common Windows checkout locations.
    bucket_dir = output / "_tmp" / tag
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    bucket_paths = [bucket_dir / f"bucket_{index:02d}.csv" for index in range(32)]
    rows_read = 0
    for chunk_number, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        rows_read += len(chunk)
        groups = chunk["split_group_id"].astype(str)
        bucket_numbers = groups.map(lambda value: zlib.crc32(value.encode("utf-8")) % len(bucket_paths)).to_numpy(dtype=int)
        for bucket_number in np.unique(bucket_numbers):
            bucket_path = bucket_paths[int(bucket_number)]
            subset = chunk.loc[bucket_numbers == bucket_number]
            subset.to_csv(bucket_path, mode="a", header=not bucket_path.exists(), index=False, encoding="utf-8")
        if chunk_number % 8 == 0:
            print(f"partitioning {path.name}: rows={rows_read:,}", flush=True)
    groups_yielded = 0
    for bucket_number, bucket_path in enumerate(bucket_paths, start=1):
        if not bucket_path.exists():
            continue
        bucket = pd.read_csv(bucket_path, low_memory=False)
        for group_id, group in bucket.groupby("split_group_id", sort=False, dropna=False):
            groups_yielded += 1
            yield str(group_id), group
        bucket_path.unlink()
        print(f"grouping {path.name}: bucket={bucket_number}/32, groups={groups_yielded}", flush=True)
    shutil.rmtree(bucket_dir, ignore_errors=True)
    print(f"read {path.name}: rows={rows_read:,}, groups={groups_yielded}", flush=True)


def fit_transaction_scaler(train_path: Path, *, output: Path) -> dict[str, dict[str, float | str]]:
    values: dict[str, list[np.ndarray]] = {feature: [] for feature in SEQUENCE_CONTINUOUS if feature != "interarrival_time"}
    raw_columns = [feature for feature in values]
    for chunk in pd.read_csv(train_path, usecols=raw_columns, chunksize=CHUNK_SIZE, low_memory=False):
        for feature in raw_columns:
            array = pd.to_numeric(chunk[feature], errors="coerce").to_numpy(dtype=np.float64)
            array = array[np.isfinite(array)]
            if feature in BINARY_FEATURES:
                array = np.clip(array, 0, 1)
            elif feature not in ENTROPY_FEATURES:
                array = np.log1p(np.clip(array, 0, None))
            values[feature].append(array)

    iat_values: list[np.ndarray] = []
    for _, group in iter_groups(train_path, ["split_group_id", "request_timestamp"], "iat_fit", output=output):
        ts = pd.to_numeric(group["request_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        ts = np.sort(ts[np.isfinite(ts)])
        if len(ts) > 1:
            iat = np.diff(ts)
            iat = np.log1p(iat[(iat >= 0) & np.isfinite(iat)])
            iat_values.append(iat)

    scaler: dict[str, dict[str, float | str]] = {}
    for feature in SEQUENCE_CONTINUOUS:
        if feature in BINARY_FEATURES:
            scaler[feature] = {"transformation": "identity_clip_0_1", "center": 0.0, "iqr": 1.0}
            continue
        parts = iat_values if feature == "interarrival_time" else values[feature]
        array = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        if array.size:
            q25, center, q75 = np.quantile(array, [0.25, 0.5, 0.75])
            iqr = float(q75 - q25)
            if not math.isfinite(iqr) or iqr <= 1e-12:
                iqr = 1.0
        else:
            center, iqr = 0.0, 1.0
        scaler[feature] = {
            "transformation": "robust" if feature in ENTROPY_FEATURES else "log1p_robust",
            "center": float(center),
            "iqr": float(iqr),
        }
    return scaler


def transform_value(value: object, feature: str, scaler: dict[str, dict[str, float | str]]) -> float:
    number = safe_float(value)
    rule = scaler[feature]
    if feature in BINARY_FEATURES:
        return float(np.clip(number, 0, 1)) if math.isfinite(number) else 0.0
    if not math.isfinite(number):
        return 0.0
    if rule["transformation"] == "log1p_robust":
        number = math.log1p(max(0.0, number))
    return float((number - float(rule["center"])) / float(rule["iqr"]))


def build_sequence(window: pd.DataFrame, scaler: dict[str, dict[str, float | str]]):
    count = len(window)
    continuous = np.zeros((WINDOW_SIZE, len(SEQUENCE_CONTINUOUS)), dtype=np.float32)
    qtype_ids = np.zeros(WINDOW_SIZE, dtype=np.int16)
    rcode_ids = np.zeros(WINDOW_SIZE, dtype=np.int16)
    mask = np.zeros(WINDOW_SIZE, dtype=np.bool_)
    timestamps = pd.to_numeric(window["request_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    iat = np.zeros(count, dtype=np.float64)
    if count > 1:
        differences = np.diff(timestamps)
        differences[~np.isfinite(differences) | (differences < 0)] = np.nan
        iat[1:] = differences
    for row_index, (_, row) in enumerate(window.iterrows()):
        for feature_index, feature in enumerate(SEQUENCE_CONTINUOUS):
            raw = iat[row_index] if feature == "interarrival_time" else row.get(feature)
            continuous[row_index, feature_index] = transform_value(raw, feature, scaler)
        qtype = str(row.get("query_type", "UNK")).upper()
        rcode = str(row.get("dns_rcode", "UNK")).upper()
        qtype_ids[row_index] = QTYPE_VOCAB.get(qtype, QTYPE_VOCAB["UNK"])
        rcode_ids[row_index] = RCODE_VOCAB.get(rcode, RCODE_VOCAB["UNK"])
        mask[row_index] = True
    return continuous, qtype_ids, rcode_ids, mask


def build_graph(window: pd.DataFrame, continuous: np.ndarray, qtype_ids: np.ndarray):
    node_lookup: dict[tuple[str, str], int] = {}
    node_type: list[int] = []
    node_transactions: list[list[int]] = []
    edge_transactions: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    type_number = {"client": 0, "server": 1, "base": 2, "subdomain": 3}

    def node(kind: str, identity: object) -> int:
        key = (kind, str(identity) if pd.notna(identity) and str(identity) else "<missing>")
        if key not in node_lookup:
            node_lookup[key] = len(node_type)
            node_type.append(type_number[kind])
            node_transactions.append([])
        return node_lookup[key]

    for tx_index, (_, row) in enumerate(window.iterrows()):
        client = node("client", row.get("client_ip"))
        server = node("server", row.get("server_ip"))
        base = node("base", row.get("base_domain"))
        sub_value = row.get("subdomain")
        participants = [client, server, base]
        for item in participants:
            node_transactions[item].append(tx_index)
        edge_transactions[(client, server, 0)].append(tx_index)
        edge_transactions[(client, base, 1)].append(tx_index)
        if pd.notna(sub_value) and str(sub_value):
            sub = node("subdomain", sub_value)
            node_transactions[sub].append(tx_index)
            edge_transactions[(base, sub, 2)].append(tx_index)

    degrees = np.zeros(len(node_type), dtype=np.float32)
    for left, right, _ in edge_transactions:
        degrees[left] += 1
        degrees[right] += 1
    denominator = max(1, len(node_type) - 1)
    node_features = np.zeros((len(node_type), 13), dtype=np.float32)
    tunnel_ids = {QTYPE_VOCAB[x] for x in ["TXT", "NULL", "PRIVATE", "CNAME"]}
    for index, tx_indices in enumerate(node_transactions):
        node_features[index, node_type[index]] = 1.0
        idx = np.asarray(tx_indices, dtype=int)
        node_features[index, 4] = len(idx) / len(window)
        node_features[index, 5] = float(np.mean(continuous[idx, 0]))
        node_features[index, 6] = float(np.mean(continuous[idx, 1]))
        node_features[index, 7] = float(np.mean(continuous[idx, 4]))
        node_features[index, 8] = float(np.mean(continuous[idx, 11]))
        node_features[index, 9] = float(np.mean(continuous[idx, 14]))
        node_features[index, 10] = degrees[index] / denominator
        node_features[index, 11] = float(np.mean(qtype_ids[idx] == QTYPE_VOCAB["A"]))
        node_features[index, 12] = float(np.mean(np.isin(qtype_ids[idx], list(tunnel_ids))))

    edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for (left, right, edge_type), tx_indices in edge_transactions.items():
        idx = np.asarray(tx_indices, dtype=int)
        feature = [0.0] * 6
        feature[edge_type] = 1.0
        feature[3] = len(idx) / len(window)
        feature[4] = float(np.mean(continuous[idx, 11]))
        feature[5] = float(np.mean(continuous[idx, 12]))
        edges.extend([(left, right), (right, left)])
        edge_features.extend([feature, feature])
    edge_index = np.asarray(edges, dtype=np.int32).T if edges else np.zeros((2, 0), dtype=np.int32)
    edge_attr = np.asarray(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 6), dtype=np.float32)
    return node_features, edge_index, edge_attr


def process_split(split: str, path: Path, scaler: dict[str, dict[str, float | str]], *, output: Path):
    print(f"processing split={split}", flush=True)
    manifest_rows: list[dict[str, object]] = []
    tabular_rows: list[dict[str, object]] = []
    sequence_cont: list[np.ndarray] = []
    sequence_qtype: list[np.ndarray] = []
    sequence_rcode: list[np.ndarray] = []
    sequence_mask: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    graph_nodes: list[np.ndarray] = []
    graph_edges: list[np.ndarray] = []
    graph_edge_attr: list[np.ndarray] = []
    node_ptr = [0]
    edge_ptr = [0]
    excluded_groups: list[dict[str, object]] = []
    transaction_total = 0
    group_total = 0

    for group_id, group in iter_groups(path, TABULAR_INPUT_COLUMNS, f"representation_{split}", output=output):
        group_total += 1
        transaction_total += len(group)
        labels_in_group = group["traffic_label"].astype(str).str.lower().unique()
        if len(labels_in_group) != 1:
            raise RuntimeError(f"Mixed labels in {group_id}: {labels_in_group}")
        group = group.assign(_timestamp=pd.to_numeric(group["request_timestamp"], errors="coerce"))
        group = group.sort_values("_timestamp", kind="mergesort").drop(columns="_timestamp").reset_index(drop=True)
        made = 0
        for start in range(0, len(group), WINDOW_SIZE):
            window = group.iloc[start:start + WINDOW_SIZE]
            if len(window) < MIN_WINDOW_SIZE:
                continue
            window_number = start // WINDOW_SIZE
            sample_id = stable_sample_id(split, group_id, window_number)
            label = 1 if labels_in_group[0] == "malicious" else 0
            tab = aggregate_window(window, window_number)
            tab["sample_id"] = sample_id
            tab["label"] = label
            tabular_rows.append(tab)
            cont, qtype, rcode, mask = build_sequence(window, scaler)
            nodes, edges, edge_attr = build_graph(window, cont, qtype)
            sequence_cont.append(cont)
            sequence_qtype.append(qtype)
            sequence_rcode.append(rcode)
            sequence_mask.append(mask)
            labels.append(label)
            sample_ids.append(sample_id)
            graph_nodes.append(nodes)
            graph_edges.append(edges + node_ptr[-1])
            graph_edge_attr.append(edge_attr)
            node_ptr.append(node_ptr[-1] + len(nodes))
            edge_ptr.append(edge_ptr[-1] + edges.shape[1])
            manifest_rows.append({
                "sample_id": sample_id,
                "split": split,
                "split_group_id": group_id,
                "source_name": str(window["source_name"].iloc[0]),
                "traffic_label": labels_in_group[0],
                "tool_label": str(window["tool_label"].iloc[0]),
                "scenario_type": str(window["scenario_type"].iloc[0]),
                "evaluation_subset": str(window["evaluation_subset"].iloc[0]),
                "window_number": window_number,
                "window_tx_count": len(window),
                "window_start_timestamp": safe_float(window["request_timestamp"].iloc[0]),
                "window_end_timestamp": safe_float(window["request_timestamp"].iloc[-1]),
                "graph_nodes": len(nodes),
                "graph_directed_edges": edges.shape[1],
            })
            made += 1
        if made == 0:
            excluded_groups.append({"split": split, "split_group_id": group_id, "transactions": len(group), "reason": "fewer_than_32_transactions"})
        if group_total % 20 == 0:
            print(f"{split}: groups={group_total}, windows={len(sample_ids):,}", flush=True)

    tabular = pd.DataFrame(tabular_rows)
    feature_columns = [column for column in tabular.columns if column not in META_COLUMNS + ["sample_id", "label"]]
    tabular = tabular[["sample_id", "label"] + feature_columns]
    split_dir = output / "prepared" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    tabular.to_csv(split_dir / f"{split}_tabular_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(split_dir / f"{split}_sample_manifest.csv", index=False, encoding="utf-8-sig")
    if excluded_groups:
        pd.DataFrame(excluded_groups).to_csv(split_dir / f"{split}_excluded_small_groups.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        split_dir / f"{split}_sequence.npz",
        continuous=np.stack(sequence_cont).astype(np.float32),
        qtype_id=np.stack(sequence_qtype).astype(np.int16),
        rcode_id=np.stack(sequence_rcode).astype(np.int16),
        mask=np.stack(sequence_mask),
        y=np.asarray(labels, dtype=np.int8),
        sample_id=np.asarray(sample_ids, dtype="U40"),
    )
    np.savez_compressed(
        split_dir / f"{split}_graph.npz",
        node_features=np.concatenate(graph_nodes, axis=0).astype(np.float32),
        edge_index=np.concatenate(graph_edges, axis=1).astype(np.int32),
        edge_features=np.concatenate(graph_edge_attr, axis=0).astype(np.float32),
        graph_node_ptr=np.asarray(node_ptr, dtype=np.int64),
        graph_edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        y=np.asarray(labels, dtype=np.int8),
        sample_id=np.asarray(sample_ids, dtype="U40"),
    )
    return tabular, {
        "split": split,
        "source_rows": transaction_total,
        "source_groups": group_total,
        "windows": len(sample_ids),
        "represented_groups": len({row["split_group_id"] for row in manifest_rows}),
        "excluded_small_groups": len(excluded_groups),
        "benign_windows": int(sum(label == 0 for label in labels)),
        "malicious_windows": int(sum(label == 1 for label in labels)),
        "graph_nodes": int(node_ptr[-1]),
        "graph_directed_edges": int(edge_ptr[-1]),
        "tabular_feature_count": len(feature_columns),
        "sequence_continuous_channels": len(SEQUENCE_CONTINUOUS),
    }, feature_columns


def tabular_rule(feature: str) -> str:
    ratio_names = ("_ratio", "_share", "jaro_winkler")
    entropy = "entropy" in feature
    if feature.endswith(ratio_names):
        return "identity_clip_0_1"
    if entropy:
        return "robust"
    return "log1p_robust"


def fit_tabular_scaler(train: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float | str]]:
    scaler: dict[str, dict[str, float | str]] = {}
    for feature in feature_columns:
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=np.float64)
        rule = tabular_rule(feature)
        if rule == "identity_clip_0_1":
            scaler[feature] = {"transformation": rule, "center": 0.0, "iqr": 1.0}
            continue
        valid = values[np.isfinite(values)]
        if rule == "log1p_robust":
            valid = np.log1p(np.clip(valid, 0, None))
        if valid.size:
            q25, center, q75 = np.quantile(valid, [0.25, 0.5, 0.75])
            iqr = float(q75 - q25)
            if not math.isfinite(iqr) or iqr <= 1e-12:
                iqr = 1.0
        else:
            center, iqr = 0.0, 1.0
        scaler[feature] = {"transformation": rule, "center": float(center), "iqr": float(iqr)}
    return scaler


def scale_tabular(frame: pd.DataFrame, feature_columns: list[str], scaler: dict[str, dict[str, float | str]]) -> pd.DataFrame:
    output = frame[["sample_id", "label"]].copy()
    for feature in feature_columns:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64)
        rule = scaler[feature]
        if rule["transformation"] == "identity_clip_0_1":
            values = np.clip(values, 0, 1)
            values[~np.isfinite(values)] = 0.0
        else:
            if rule["transformation"] == "log1p_robust":
                values = np.log1p(np.clip(values, 0, None))
            values = (values - float(rule["center"])) / float(rule["iqr"])
            values[~np.isfinite(values)] = 0.0
        output[feature] = values.astype(np.float32)
    return output


def write_specs(transaction_scaler, tabular_scaler, feature_columns, *, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "transaction_scaling_parameters.json").write_text(json.dumps(transaction_scaler, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"feature": feature, **values, "fit_partition": "train_only"}
        for feature, values in tabular_scaler.items()
    ]).to_csv(output / "tabular_scaling_parameters.csv", index=False, encoding="utf-8-sig")
    representation_spec = {
        "sample_unit": {"window_size": WINDOW_SIZE, "minimum_partial_window": MIN_WINDOW_SIZE, "overlap": 0, "boundary": "split_group_id", "max_windows_per_group": None},
        "tabular": {"features": feature_columns, "feature_count": len(feature_columns), "raw_identity_fields_in_predictors": []},
        "sequence": {"shape": [WINDOW_SIZE, len(SEQUENCE_CONTINUOUS)], "continuous_channels": SEQUENCE_CONTINUOUS, "categorical_channels": ["qtype_id", "rcode_id"], "padding_mask": True, "qtype_vocab": QTYPE_VOCAB, "rcode_vocab": RCODE_VOCAB},
        "graph": {
            "node_types": ["client", "dns_server", "base_domain", "subdomain"],
            "edge_types": ["client_dns_server", "client_base_domain", "base_domain_subdomain"],
            "node_features": ["type_client", "type_dns_server", "type_base_domain", "type_subdomain", "activity_ratio", "mean_scaled_query_name_len", "mean_scaled_subdomain_len", "mean_scaled_query_entropy", "mean_scaled_transaction_bytes", "response_missing_ratio", "degree_ratio", "qtype_a_ratio", "tunnel_associated_qtype_ratio"],
            "edge_features": ["type_client_dns_server", "type_client_base_domain", "type_base_domain_subdomain", "interaction_ratio", "mean_scaled_transaction_bytes", "mean_scaled_response_time"],
            "raw_identity_fields_in_predictors": [],
            "note": "Raw endpoint/domain strings are used only to construct within-window topology and are not stored as model features. This is a project proxy graph because recursive resolver paths from GraphTunnel are unavailable in the unified schema.",
        },
    }
    (output / "representation_specification.json").write_text(json.dumps(representation_spec, indent=2, ensure_ascii=False), encoding="utf-8")
