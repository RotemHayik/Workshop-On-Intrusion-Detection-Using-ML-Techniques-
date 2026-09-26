"""Train-only fitting, Validation selection, and checkpointed experiment runs."""

import argparse, json, time, random, copy, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from joblib import dump
from .data import (
    load_tabular,
    recording_weights,
    SequenceData,
    GraphData,
    graph_collate,
)
from .models import BiLSTM, GraphSAGE
from .metrics import choose_threshold, metrics, aggregate_recordings


def seed_everything(seed, threads=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predictions_frame(manifest, y, score):
    frame = manifest.copy()
    frame["label"] = y
    frame["score"] = score
    return frame


def save_validation(path, frame):
    threshold = choose_threshold(frame.label, frame.score)
    frame["prediction"] = (frame.score >= threshold).astype(int)
    frame.to_csv(path / "validation_predictions.csv", index=False)
    rec = aggregate_recordings(frame)
    rt = choose_threshold(rec.label, rec.score)
    summary = {
        "window": metrics(frame.label, frame.score, threshold),
        "recording": metrics(rec.label, rec.score, rt),
    }
    (path / "validation_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def train_tree(root, config, path):
    kind = config["model"]
    p = config["parameters"]
    seed = config["seed"]
    x, y, manifest, features = load_tabular(
        root, "train", scaled=kind == "if", drop=config.get("drop_features", [])
    )
    xv, yv, mv, _ = load_tabular(
        root, "validation", scaled=kind == "if", drop=config.get("drop_features", [])
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(x)
    xv = imputer.transform(xv)
    if kind == "rf":
        model = RandomForestClassifier(**p, random_state=seed, n_jobs=3)
        weights = (
            recording_weights(manifest, y)
            if config.get("balance", "recording_class") == "recording_class"
            else None
        )
        model.fit(x, y, sample_weight=weights)
        score = model.predict_proba(xv)[:, 1]
    else:
        rng = np.random.default_rng(seed)
        indices = []
        for group in manifest.loc[y == 0, "split_group_id"].unique():
            pool = np.flatnonzero(
                (manifest.split_group_id.to_numpy() == group) & (y == 0)
            )
            indices.extend(rng.choice(pool, size=64, replace=len(pool) < 64))
        model = IsolationForest(**p, random_state=seed, n_jobs=3)
        model.fit(x[indices])
        score = -model.score_samples(xv)
        pd.DataFrame({"train_index": indices}).to_csv(
            path / "benign_training_indices.csv", index=False
        )
    dump(
        {"model": model, "imputer": imputer, "features": features, "kind": kind},
        path / "model.joblib",
        compress=3,
    )
    return predictions_frame(mv, yv, score)


def make_loaders(root, kind, config):
    cls = SequenceData if kind == "bilstm" else GraphData
    train = cls(root, "train")
    val = cls(root, "validation")
    weights = recording_weights(train.manifest, train.y.numpy())
    gen = torch.Generator().manual_seed(config["seed"])
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        len(train),
        replacement=True,
        generator=gen,
    )
    kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": 0,
        "collate_fn": graph_collate if kind == "sage" else None,
    }
    return (
        train,
        val,
        DataLoader(train, sampler=sampler, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
    )


def predict_neural(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.append(torch.sigmoid(model(*batch[:-1])).numpy())
    return np.concatenate(scores)


def train_neural(root, config, path):
    kind = config["model"]
    params = config["parameters"]
    train, val, train_loader, val_loader = make_loaders(root, kind, config)
    model = (BiLSTM if kind == "bilstm" else GraphSAGE)(**params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    criterion = nn.BCEWithLogitsLoss()
    best = -float("inf")
    stale = 0
    history = []
    best_state = None
    for epoch in range(1, config["max_epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = criterion(logits, batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += loss.item() * len(batch[-1])
            count += len(batch[-1])
        score = predict_neural(model, val_loader)
        threshold = choose_threshold(val.y.numpy(), score)
        met = metrics(val.y.numpy(), score, threshold)
        # Smooth AUPRC for epoch selection; operating threshold remains validation-only.
        criterion_value = met["auprc"]
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / count,
                "seconds": time.perf_counter() - start,
                **met,
            }
        )
        pd.DataFrame(history).to_csv(path / "learning_curve.csv", index=False)
        print(
            json.dumps(
                {
                    "run": config["run_id"],
                    "epoch": epoch,
                    "loss": round(loss_sum / count, 5),
                    "val_recall": round(met["recall"], 5),
                    "val_auprc": round(met["auprc"], 5),
                    "seconds": round(history[-1]["seconds"], 2),
                }
            ),
            flush=True,
        )
        if criterion_value > best + config["min_delta"]:
            best = criterion_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {"state_dict": best_state, "config": config, "epoch": epoch},
                path / "model.pt",
            )
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    score = predict_neural(model, val_loader)
    return predictions_frame(val.manifest, val.y.numpy().astype(int), score)


def run(root, config):
    path = Path(root) / "06_results/runs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    if (path / "COMPLETE.json").exists():
        print("Already complete", config["run_id"], flush=True)
        return
    seed_everything(config["seed"])
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.perf_counter()
    print("START", config["run_id"], flush=True)
    frame = (
        train_tree(root, config, path)
        if config["model"] in ["rf", "if"]
        else train_neural(root, config, path)
    )
    summary = save_validation(path, frame)
    result = {
        "run_id": config["run_id"],
        "model": config["model"],
        "seed": config["seed"],
        "seconds": time.perf_counter() - start,
        "validation": summary,
    }
    (path / "COMPLETE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(result), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    configs = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for conf in configs:
        if args.run_id is None or conf["run_id"] == args.run_id:
            run(args.root, conf)


if __name__ == "__main__":
    main()
