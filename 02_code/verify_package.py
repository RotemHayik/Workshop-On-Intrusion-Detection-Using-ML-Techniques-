"""Verify artifacts, locked checkpoints and representation contracts without fitting."""

import json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd


def digest(file):
    h = hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    root = Path(root)
    checks = []

    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    inventory = json.loads(
        (root / "09_reproducibility/package_inventory.json").read_text()
    )
    for item in inventory:
        file = root / item["file"]
        check(
            "sha256 " + item["file"],
            file.is_file()
            and file.stat().st_size == item["bytes"]
            and digest(file) == item["sha256"],
        )
    groups = {}
    for split in ["train", "validation", "test", "external_test"]:
        folder = root / "04_data/prepared" / split
        manifest = pd.read_csv(folder / f"{split}_sample_manifest.csv")
        raw = pd.read_csv(folder / f"{split}_tabular_core42_raw.csv")
        scaled = pd.read_csv(folder / f"{split}_tabular_core42_scaled.csv")
        seq = np.load(folder / f"{split}_sequence.npz")
        graph = np.load(folder / f"{split}_graph.npz")
        ids = manifest.sample_id.to_numpy()
        check(split + " unique IDs", manifest.sample_id.is_unique)
        for label, arr in [
            ("raw", raw.sample_id.to_numpy()),
            ("scaled", scaled.sample_id.to_numpy()),
            ("sequence", seq["sample_id"]),
            ("graph", graph["sample_id"]),
        ]:
            check(split + " " + label + " ordered IDs", np.array_equal(ids, arr))
        check(
            split + " 42 feature contract",
            len(raw.columns) == 44 and raw.columns.equals(scaled.columns),
        )
        y = manifest.traffic_label.eq("malicious").to_numpy(dtype=int)
        for label, arr in [
            ("raw", raw.label),
            ("sequence", seq["y"]),
            ("graph", graph["y"]),
        ]:
            check(split + " " + label + " aligned labels", np.array_equal(y, arr))
        check(
            split + " mask",
            np.array_equal(seq["mask"].sum(1), manifest.window_tx_count),
        )
        check(
            split + " finite arrays",
            np.isfinite(scaled.iloc[:, 2:]).all().all()
            and np.isfinite(seq["continuous"]).all()
            and np.isfinite(graph["node_features"]).all(),
        )
        # NPZ members decompress on access; load each once before checking windows.
        node_ptr = graph["graph_node_ptr"]
        edge_ptr = graph["graph_edge_ptr"]
        edge_index = graph["edge_index"]
        check(
            split + " pointer lengths",
            len(node_ptr) == len(ids) + 1 and len(edge_ptr) == len(ids) + 1,
        )
        check(
            split + " pointer coverage",
            node_ptr[0] == 0
            and edge_ptr[0] == 0
            and node_ptr[-1] == len(graph["node_features"])
            and edge_ptr[-1] == edge_index.shape[1]
            and (np.diff(node_ptr) >= 0).all()
            and (np.diff(edge_ptr) >= 0).all(),
        )
        for i in range(len(ids)):
            lo, hi = node_ptr[i : i + 2]
            a, b = edge_ptr[i : i + 2]
            edges = edge_index[:, a:b]
            if edges.size and not ((edges >= lo).all() and (edges < hi).all()):
                raise AssertionError("Cross-window graph edge")
        check(split + " graph boundaries", True)
        groups[split] = set(manifest.split_group_id)
    for i, left in enumerate(groups):
        for right in list(groups)[i + 1 :]:
            check(
                left + " / " + right + " recording disjointness",
                not groups[left] & groups[right],
            )
    lock = json.loads((root / "03_config/selection_lock.json").read_text())
    for file, expected in lock["sha256"].items():
        check(
            "selection lock " + file,
            digest(root / Path(file.replace("\\", "/"))) == expected,
        )
    check(
        "22 training runs",
        len(list((root / "06_results/runs").glob("*/COMPLETE.json"))) == 22,
    )
    for split in ["validation", "test", "external_test"]:
        summary = json.loads(
            (root / "06_results/cascade" / split / "summary.json").read_text()
        )
        pred = pd.read_csv(root / "06_results/cascade" / split / "predictions.csv")
        check(split + " cascade population", summary["n"] == len(pred))
        check(
            split + " routing accounted for",
            summary["stage_two"]
            == summary["llm_requested"] + summary["budget_fallback"],
        )
        check(
            split + " responses accounted for",
            summary["llm_requested"]
            == summary["llm_valid"] + summary["invalid_fallback"],
        )
        if split == "external_test":
            check(
                "external metrics unavailable",
                summary["metrics"]["arbitrated"]["window"]["f1"] is None,
            )
    return {"all_passed": True, "checks_passed": len(checks), "checks": checks}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--save", type=Path)
    a = p.parse_args()
    result = verify(a.root)
    if a.save:
        a.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
