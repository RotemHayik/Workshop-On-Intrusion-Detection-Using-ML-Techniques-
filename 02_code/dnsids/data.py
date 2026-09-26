"""Dataset-agnostic loaders for the frozen, aligned Part 2 representations."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

META = {"sample_id", "label"}


def load_tabular(root, split, scaled=False, drop=()):
    folder = Path(root) / "04_data/prepared" / split
    frame = pd.read_csv(
        folder / f'{split}_tabular_core42_{"scaled" if scaled else "raw"}.csv'
    )
    manifest = pd.read_csv(folder / f"{split}_sample_manifest.csv")
    if not np.array_equal(frame.sample_id, manifest.sample_id):
        raise ValueError("Tabular/manifest order mismatch")
    features = [c for c in frame if c not in META and c not in drop]
    return (
        frame[features].to_numpy(dtype=np.float32),
        frame.label.to_numpy(dtype=np.int64),
        manifest,
        features,
    )


def recording_weights(manifest, y):
    """Equal total mass per class and per recording within each class."""
    groups = manifest.split_group_id.astype(str).to_numpy()
    w = np.zeros(len(y), dtype=np.float64)
    for label in np.unique(y):
        ids = np.unique(groups[y == label])
        for group in ids:
            mask = (groups == group) & (y == label)
            w[mask] = 1 / (len(ids) * mask.sum())
    return w / w.mean()


class SequenceData(Dataset):
    def __init__(self, root, split):
        p = Path(root) / "04_data/prepared" / split
        with np.load(p / f"{split}_sequence.npz", allow_pickle=False) as z:
            self.x = torch.from_numpy(z["continuous"].copy())
            self.q = torch.from_numpy(z["qtype_id"].astype(np.int64))
            self.r = torch.from_numpy(z["rcode_id"].astype(np.int64))
            self.length = torch.from_numpy(z["mask"].sum(1).astype(np.int64))
            self.y = torch.from_numpy(z["y"].astype(np.float32))
            ids = z["sample_id"].astype(str)
        self.manifest = pd.read_csv(p / f"{split}_sample_manifest.csv")
        if not np.array_equal(ids, self.manifest.sample_id):
            raise ValueError("Sequence alignment")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.q[i], self.r[i], self.length[i], self.y[i]


class GraphData(Dataset):
    def __init__(self, root, split):
        p = Path(root) / "04_data/prepared" / split
        with np.load(p / f"{split}_graph.npz", allow_pickle=False) as z:
            self.x = torch.from_numpy(z["node_features"].copy())
            self.edges = torch.from_numpy(z["edge_index"].astype(np.int64))
            self.nptr = z["graph_node_ptr"].copy()
            self.eptr = z["graph_edge_ptr"].copy()
            self.y = torch.from_numpy(z["y"].astype(np.float32))
            ids = z["sample_id"].astype(str)
        self.manifest = pd.read_csv(p / f"{split}_sample_manifest.csv")
        if not np.array_equal(ids, self.manifest.sample_id):
            raise ValueError("Graph alignment")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a, b = self.nptr[i : i + 2]
        c, d = self.eptr[i : i + 2]
        return self.x[a:b], self.edges[:, c:d] - int(a), self.y[i]


def graph_collate(items):
    xs = []
    edges = []
    batch = []
    ys = []
    offset = 0
    for i, (x, e, y) in enumerate(items):
        xs.append(x)
        edges.append(e + offset)
        batch.append(torch.full((len(x),), i, dtype=torch.long))
        ys.append(y)
        offset += len(x)
    return torch.cat(xs), torch.cat(edges, 1), torch.cat(batch), torch.stack(ys)
