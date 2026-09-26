"""Freeze model choices using completed Validation experiments only."""

import argparse, json, hashlib, datetime, copy
from pathlib import Path
import pandas as pd


def select(root):
    root = Path(root)
    files = list((root / "06_results/runs").glob("*/COMPLETE.json"))
    results = [
        json.loads(p.read_text())
        for p in files
        if not p.parent.name.endswith(("_seed17", "_seed2026"))
    ]
    expected = json.loads(
        (root / "03_config/experiments.json").read_text()
    ) + json.loads((root / "03_config/refinements.json").read_text())
    if {c["run_id"] for c in expected} != {c["run_id"] for c in results}:
        raise RuntimeError("Not all declared Validation runs are complete")
    # Exact metric ties favor the base; candidate order is the predeclared order.
    order = {c["run_id"]: i for i, c in enumerate(expected)}
    key = lambda r: (
        r["validation"]["window"]["recall"],
        -r["validation"]["window"]["fpr"],
        -order[r["run_id"]],
    )
    selected = {
        kind: max([r for r in results if r["model"] == kind], key=key)["run_id"]
        for kind in ["rf", "if", "bilstm", "sage"]
    }
    hashes = {}
    for name in selected.values():
        folder = root / "06_results/runs" / name
        for file in folder.glob("*"):
            if file.suffix in [".joblib", ".pt"] or file.name in [
                "config.json",
                "validation_metrics.json",
            ]:
                hashes[str(file.relative_to(root))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    lock = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection_rule": "Highest Validation recall at FPR <= 1%, then lower FPR, then earlier predeclared candidate; never choose a random seed on performance",
        "selected": selected,
        "evaluate_runs": list(
            dict.fromkeys(
                ["rf_base", "if_base", "bilstm_base", "sage_base"]
                + list(selected.values())
            )
        ),
        "sha256": hashes,
    }
    destination = root / "03_config/selection_lock.json"
    if destination.exists():
        raise FileExistsError("Selection is already frozen")
    destination.write_text(json.dumps(lock, indent=2))
    rows = []
    for r in results:
        rows.append(
            {
                "run_id": r["run_id"],
                "model": r["model"],
                "seed": r["seed"],
                "training_seconds": r["seconds"],
                **r["validation"]["window"],
            }
        )
    pd.DataFrame(rows).to_csv(
        root / "06_results/validation_sensitivity.csv", index=False
    )
    repeats = []
    for kind in ["bilstm", "sage"]:
        conf = json.loads(
            (root / "06_results/runs" / selected[kind] / "config.json").read_text()
        )
        for seed in [17, 2026]:
            c = copy.deepcopy(conf)
            c["seed"] = seed
            c["run_id"] = conf["run_id"] + "_seed" + str(seed)
            repeats.append(c)
    (root / "03_config/seed_repeats.json").write_text(json.dumps(repeats, indent=2))
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    select(a.root)
