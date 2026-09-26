"""Behavioral cascade, bounded local LLM arbitration, and full-population audit."""

import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from .metrics import metrics, aggregate_recordings, choose_threshold, cluster_intervals

FEATURES = [
    "query_name_len_mean",
    "subdomain_len_mean",
    "query_entropy_mean",
    "longest_subdomain_entropy_mean",
    "transaction_bytes_mean",
    "top_base_domain_share",
    "unique_subdomain_ratio",
    "repeated_query_ratio",
    "qtype_a_ratio",
    "qtype_txt_ratio",
]
SYSTEM = """You assess DNS tunneling from a completed traffic window. Use only the supplied numeric observations, Train reference medians and detector decisions. Anomaly alone is not proof of attack: legitimate background traffic can have large responses. Combine lexical payload, domain concentration, recurrence and query-type evidence. No labels, source identities or tool identities are provided. Return a JSON object with label (benign or malicious), evidence (one short sentence naming the observed features), and limitation (one short uncertainty statement). Do not invent packet content or reveal a hidden reasoning trace."""


def assemble(root, split, selected):
    joined = None
    for kind, name in selected.items():
        file = (
            root / "06_results/runs" / name / "validation_predictions.csv"
            if split == "validation"
            else root / "06_results/final" / name / f"{split}_predictions.csv"
        )
        f = pd.read_csv(file)
        if joined is None:
            joined = f.drop(columns=["score", "prediction"])
        joined = joined.merge(
            f[["sample_id", "score", "prediction"]].rename(
                columns={"score": kind + "_score", "prediction": kind + "_prediction"}
            ),
            on="sample_id",
            validate="one_to_one",
        )
    return joined


def route(frame, policy):
    frame = frame.copy()
    clear = (frame.rf_score <= 0.2) | (frame.rf_score >= 0.8)
    agree = frame.rf_prediction == frame.if_prediction
    frame["stage_two"] = ~(clear & agree)
    vote = (
        frame[["rf_prediction", "bilstm_prediction", "sage_prediction"]].sum(axis=1)
        >= 2
    ).astype(int)
    frame["cascade_prediction"] = np.where(frame.stage_two, vote, frame.rf_prediction)
    candidates = frame[frame.stage_two][["sample_id", "split_group_id"]].copy()
    candidates["order"] = candidates.sample_id.map(
        lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    candidates = (
        candidates.sort_values("order")
        .groupby("split_group_id", sort=False)
        .head(policy["llm_budget_per_recording"])
        .sort_values("order")
        .head(policy["llm_budget_per_split"])
    )
    frame["llm_requested"] = frame.sample_id.isin(candidates.sample_id)
    return frame


def train_reference(root):
    f = pd.read_csv(root / "04_data/prepared/train/train_tabular_core42_raw.csv")
    m = pd.read_csv(root / "04_data/prepared/train/train_sample_manifest.csv")
    # Equal recording influence: median of recording medians, separately by class.
    g = (
        f.merge(m[["sample_id", "split_group_id"]], on="sample_id")
        .groupby(["label", "split_group_id"])[FEATURES]
        .median()
    )
    return {
        ("benign" if label == 0 else "malicious"): {
            k: round(float(v), 4) for k, v in part.median().items()
        }
        for label, part in g.groupby(level=0)
    }


def prompt_for(row, refs):
    return {
        "window_features": {
            f: round(float(row[f]), 4) if np.isfinite(row[f]) else None
            for f in FEATURES
        },
        "train_recording_median_reference": refs,
        "detector_decisions": {
            k: ("malicious" if row[k + "_prediction"] else "benign")
            for k in ["rf", "if", "bilstm", "sage"]
        },
        "rf_score": round(float(row.rf_score), 4),
    }


def parse_response(raw):
    obj = json.loads(raw)
    if obj.get("label") not in ["benign", "malicious"]:
        raise ValueError("Unsupported label")
    if not all(
        isinstance(obj.get(k), str) and obj[k].strip()
        for k in ["evidence", "limitation"]
    ):
        raise ValueError("Missing evidence or limitation")
    return obj


def run(root, split, endpoint="http://127.0.0.1:8765"):
    root = Path(root)
    policy = json.loads((root / "03_config/cascade_policy.json").read_text())
    selected = json.loads((root / "03_config/selection_lock.json").read_text())[
        "selected"
    ]
    folder = root / "06_results/cascade" / split
    folder.mkdir(parents=True, exist_ok=True)
    frame = route(assemble(root, split, selected), policy)
    features = pd.read_csv(
        root / "04_data/prepared" / split / f"{split}_tabular_core42_raw.csv"
    )[["sample_id"] + FEATURES]
    merged = frame.merge(features, on="sample_id", validate="one_to_one")
    refs = train_reference(root)
    logs = folder / "llm_responses.jsonl"
    completed = {}
    if logs.exists():
        for line in logs.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed[item["sample_id"]] = item
    (folder / "system_prompt.txt").write_text(SYSTEM, encoding="utf-8")
    for _, row in merged[merged.llm_requested].iterrows():
        if row.sample_id in completed:
            continue
        prompt = prompt_for(row, refs)
        start = time.perf_counter()
        item = {"sample_id": row.sample_id, "prompt": prompt}
        try:
            payload = {
                "model": "local",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                "temperature": 0,
                "seed": 42,
                "max_tokens": policy["llm"]["max_tokens"],
                "response_format": {"type": "json_object"},
            }
            response = requests.post(
                endpoint + "/v1/chat/completions",
                json=payload,
                timeout=policy["llm"]["request_timeout_seconds"],
            )
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            item["raw_response"] = raw
            item["parsed"] = parse_response(raw)
            item["usage"] = body.get("usage")
            item["valid"] = True
        except Exception as exc:
            item.update(valid=False, error=type(exc).__name__ + ": " + str(exc))
        item["seconds"] = time.perf_counter() - start
        with logs.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        completed[row.sample_id] = item
        print(
            "LLM",
            split,
            len(completed),
            item["valid"],
            round(item["seconds"], 2),
            flush=True,
        )
    frame["llm_valid"] = frame.sample_id.map(
        lambda s: completed.get(s, {}).get("valid", False)
    )
    frame["arbitrated_prediction"] = [
        (
            int(completed[s]["parsed"]["label"] == "malicious")
            if completed.get(s, {}).get("valid")
            else int(base)
        )
        for s, base in zip(frame.sample_id, frame.cascade_prediction)
    ]
    for name in ["cascade", "arbitrated"]:
        frame[name + "_score"] = frame[name + "_prediction"].astype(float)
    result = {
        "split": split,
        "n": len(frame),
        "stage_two": int(frame.stage_two.sum()),
        "llm_requested": int(frame.llm_requested.sum()),
        "llm_valid": int(frame.llm_valid.sum()),
        "budget_fallback": int((frame.stage_two & ~frame.llm_requested).sum()),
        "invalid_fallback": int((frame.llm_requested & ~frame.llm_valid).sum()),
        "llm_seconds": sum(x["seconds"] for x in completed.values()),
        "metrics": {},
    }
    lockpath = root / "03_config/cascade_validation_lock.json"
    if split == "validation":
        before = metrics(frame.label, frame.cascade_score, 0.5)
        after = metrics(frame.label, frame.arbitrated_score, 0.5)
        enabled = bool(
            result["llm_valid"] > 0
            and after["fpr"] <= 0.01
            and after["f1"] >= before["f1"]
        )
        thresholds = {
            name: choose_threshold(
                aggregate_recordings(frame.assign(score=frame[name + "_score"])).label,
                aggregate_recordings(frame.assign(score=frame[name + "_score"])).score,
            )
            for name in ["cascade", "arbitrated"]
        }
        lock = {
            "llm_overrides_enabled": enabled,
            "recording_thresholds": thresholds,
            "validation_before": before,
            "validation_after": after,
        }
        if lockpath.exists() and json.loads(lockpath.read_text()) != lock:
            raise RuntimeError("Validation cascade lock changed")
        lockpath.write_text(json.dumps(lock, indent=2))
    else:
        lock = json.loads(lockpath.read_text())
    frame["production_prediction"] = (
        frame.arbitrated_prediction
        if lock["llm_overrides_enabled"]
        else frame.cascade_prediction
    )
    for name in ["cascade", "arbitrated", "production"]:
        score = frame[name + "_prediction"].astype(float)
        f = frame.assign(score=score)
        rt = lock["recording_thresholds"][
            (
                "arbitrated"
                if name == "production" and lock["llm_overrides_enabled"]
                else ("cascade" if name == "production" else name)
            )
        ]
        rec = aggregate_recordings(f)
        result["metrics"][name] = {
            "window": metrics(f.label, f.score, 0.5),
            "recording": metrics(rec.label, rec.score, rt),
            "window_ci": cluster_intervals(f, 0.5),
        }
    for subset, condition in [
        ("requested", frame.llm_requested),
        ("valid", frame.llm_valid),
    ]:
        f = frame[condition]
        result[subset + "_edge_comparison"] = {
            "n": len(f),
            "before_correct": int((f.cascade_prediction == f.label).sum()),
            "after_correct": int((f.arbitrated_prediction == f.label).sum()),
            "corrected": int(
                (
                    (f.cascade_prediction != f.label)
                    & (f.arbitrated_prediction == f.label)
                ).sum()
            ),
            "harmed": int(
                (
                    (f.cascade_prediction == f.label)
                    & (f.arbitrated_prediction != f.label)
                ).sum()
            ),
        }
    frame.to_csv(folder / "predictions.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CASCADE COMPLETE", split, result["requested_edge_comparison"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--split", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:8765")
    a = p.parse_args()
    run(a.root, a.split, a.endpoint)
