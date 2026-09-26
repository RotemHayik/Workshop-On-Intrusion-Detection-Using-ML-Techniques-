"""Operating thresholds, binary metrics, and recording aggregation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
)


def choose_threshold(y, score, max_fpr=0.01):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both Validation classes")
    fpr, tpr, th = roc_curve(y, score, drop_intermediate=False)
    th[~np.isfinite(th)] = np.nextafter(score.max(), np.inf)
    candidates = np.where(fpr <= max_fpr + 1e-12)[0]
    # Maximum recall; then fewer false positives; then stricter threshold.
    best = max(candidates, key=lambda i: (tpr[i], -fpr[i], th[i]))
    return float(th[best])


def metrics(y, score, threshold, pred=None):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (
        (score >= threshold).astype(int)
        if pred is None
        else np.asarray(pred, dtype=int)
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "n": len(y),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }
    if tn + fp and tp + fn:
        precision = tp / (tp + fp) if tp + fp else 0.0
        result.update(
            accuracy=float((tp + tn) / len(y)),
            precision=float(precision),
            f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            fpr=float(fp / (fp + tn)),
            specificity=float(tn / (fp + tn)),
            balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2),
            auroc=float(roc_auc_score(y, score)),
            auprc=float(average_precision_score(y, score)),
            mcc=float(matthews_corrcoef(y, pred)),
        )
    else:
        result.update(
            {
                k: None
                for k in [
                    "accuracy",
                    "precision",
                    "f1",
                    "fpr",
                    "specificity",
                    "balanced_accuracy",
                    "auroc",
                    "auprc",
                    "mcc",
                ]
            }
        )
        if tn + fp:
            result["fpr"] = float(fp / (tn + fp))
            result["specificity"] = float(tn / (tn + fp))
    return result


def aggregate_recordings(frame):
    return (
        frame.groupby("split_group_id", sort=True)
        .agg(
            label=("label", "first"),
            score=("score", lambda s: float(np.quantile(s, 0.95))),
            source_name=("source_name", "first"),
        )
        .reset_index()
    )


def cluster_intervals(frame, threshold, repeats=500, seed=20260905):
    """Stratified recording bootstrap; windows never resampled independently."""
    work = frame.copy()
    work["pred"] = (work.score >= threshold).astype(int)
    work["tp"] = ((work.label == 1) & (work.pred == 1)).astype(int)
    work["fn"] = ((work.label == 1) & (work.pred == 0)).astype(int)
    work["tn"] = ((work.label == 0) & (work.pred == 0)).astype(int)
    work["fp"] = ((work.label == 0) & (work.pred == 1)).astype(int)
    g = work.groupby("split_group_id").agg(
        label=("label", "first"),
        tp=("tp", "sum"),
        tn=("tn", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
    )
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ["recall", "fpr", "precision", "f1", "accuracy"]}
    arrays = [
        g[g.label == c][["tp", "tn", "fp", "fn"]].to_numpy()
        for c in [0, 1]
        if (g.label == c).any()
    ]
    for _ in range(repeats):
        tp, tn, fp, fn = sum(
            (a[rng.integers(len(a), size=len(a))].sum(0) for a in arrays)
        )
        values["recall"].append(tp / (tp + fn) if tp + fn else np.nan)
        if tn + fp:
            values["fpr"].append(fp / (fp + tn))
        if tn + fp and tp + fn:
            values["precision"].append(tp / (tp + fp) if tp + fp else 0)
            values["f1"].append(2 * tp / (2 * tp + fp + fn))
            values["accuracy"].append((tp + tn) / (tp + tn + fp + fn))
    return {
        k: (
            np.quantile(v, [0.025, 0.975]).tolist()
            if len(v) and np.isfinite(v).all()
            else None
        )
        for k, v in values.items()
    }
