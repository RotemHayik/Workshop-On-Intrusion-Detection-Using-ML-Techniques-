"""Exercise the complete input-building path using a small real-data fixture."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnsids.prepare import build

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test", "external_test")


@pytest.fixture
def canonical_inputs(tmp_path):
    folder = ROOT / "04_data/raw_examples/validation"
    sid = pd.read_csv(folder / "case_index.csv").iloc[0].sample_id
    original = pd.read_csv(folder / f"{sid}.csv")
    # A complete window, an accepted partial window, and a discarded recording.
    large = pd.concat([original, original], ignore_index=True).iloc[:160].copy()
    large["request_timestamp"] = np.arange(len(large), dtype=float)
    small = large.iloc[:31].copy()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for split in SPLITS:
        first = large.copy()
        first["split_group_id"] = f"{split}-large"
        first["traffic_label"] = "benign"
        second = small.copy()
        second["split_group_id"] = f"{split}-small"
        second["traffic_label"] = "malicious"
        pd.concat([first, second], ignore_index=True).to_csv(
            inputs / f"{split}_transactions.csv", index=False
        )
    return inputs


def test_complete_build_is_portable_and_fits_train_only(canonical_inputs, tmp_path):
    first = tmp_path / "first"
    build(canonical_inputs, first, ROOT)
    for split in SPLITS:
        folder = first / "prepared" / split
        manifest = pd.read_csv(folder / f"{split}_sample_manifest.csv")
        raw = pd.read_csv(folder / f"{split}_tabular_core42_raw.csv")
        scaled = pd.read_csv(folder / f"{split}_tabular_core42_scaled.csv")
        excluded = pd.read_csv(folder / f"{split}_excluded_small_groups.csv")
        assert manifest.window_tx_count.tolist() == [128, 32]
        assert excluded.transactions.tolist() == [31]
        assert len(raw.columns) == 44
        assert raw.columns.equals(scaled.columns)
        assert raw.sample_id.tolist() == manifest.sample_id.tolist()
        assert np.isfinite(scaled.iloc[:, 2:].to_numpy()).all()
        with np.load(folder / f"{split}_sequence.npz") as sequence:
            np.testing.assert_array_equal(sequence["sample_id"], raw.sample_id)
            np.testing.assert_array_equal(sequence["y"], raw.label)
            np.testing.assert_array_equal(sequence["mask"].sum(1), [128, 32])
            assert np.isfinite(sequence["continuous"]).all()
        with np.load(folder / f"{split}_graph.npz") as graph:
            np.testing.assert_array_equal(graph["sample_id"], raw.sample_id)
            np.testing.assert_array_equal(graph["y"], raw.label)
            for i in range(len(raw)):
                lo, hi = graph["graph_node_ptr"][i : i + 2]
                start, end = graph["graph_edge_ptr"][i : i + 2]
                edges = graph["edge_index"][:, start:end]
                assert ((edges >= lo) & (edges < hi)).all()

    # Changing only held-out observations must not change learned transforms.
    for split in SPLITS[1:]:
        file = canonical_inputs / f"{split}_transactions.csv"
        frame = pd.read_csv(file)
        frame["query_name_len"] = 10000.0
        frame.to_csv(file, index=False)
    second = tmp_path / "second"
    build(canonical_inputs, second, ROOT)
    for name in ("transaction_scaling_parameters.json", "tabular_scaling_parameters.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    one = pd.read_csv(first / "prepared/train/train_tabular_scaled.csv")
    two = pd.read_csv(second / "prepared/train/train_tabular_scaled.csv")
    pd.testing.assert_frame_equal(one, two, check_exact=True)
    assert json.loads((first / "preparation_summary.json").read_text())[0]["windows"] == 2


def test_build_rejects_recording_overlap(canonical_inputs, tmp_path):
    file = canonical_inputs / "validation_transactions.csv"
    frame = pd.read_csv(file)
    frame["split_group_id"] = frame.split_group_id.str.replace("validation", "train")
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="Recording leakage"):
        build(canonical_inputs, tmp_path / "overlap", ROOT)


@pytest.mark.parametrize("location", ["existing", "inside-inputs"])
def test_build_protects_inputs_and_existing_outputs(canonical_inputs, tmp_path, location):
    if location == "existing":
        output = tmp_path / "existing"
        output.mkdir()
        expected = FileExistsError
    else:
        output = canonical_inputs / "new-output"
        expected = ValueError
    with pytest.raises(expected):
        build(canonical_inputs, output, ROOT)
