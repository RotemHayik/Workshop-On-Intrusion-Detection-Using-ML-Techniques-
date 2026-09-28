import json
from pathlib import Path
import numpy as np
import pandas as pd
from dnsids.prepare import ingest
from dnsids import representations as ref
from dnsids.data import recording_weights

ROOT = Path(__file__).resolve().parents[2]


def test_real_raw_window_reconstructs_all_model_representations():
    folder = ROOT / "04_data/raw_examples/validation"
    row = pd.read_csv(folder / "case_index.csv").iloc[0]
    sid = row.sample_id
    raw = pd.read_csv(folder / f"{sid}.csv")
    spec = json.loads(
        (ROOT / "04_data/transaction_scaling_parameters.json").read_text()
    )
    cont, qtype, rcode, mask = ref.build_sequence(raw, spec)
    seq = np.load(ROOT / "04_data/prepared/validation/validation_sequence.npz")
    idx = list(seq["sample_id"]).index(sid)
    for calculated, field in [
        (cont, "continuous"),
        (qtype, "qtype_id"),
        (rcode, "rcode_id"),
        (mask, "mask"),
    ]:
        np.testing.assert_allclose(calculated, seq[field][idx], rtol=1e-5, atol=1e-6)
    nodes, edges, attrs = ref.build_graph(raw, cont, qtype)
    graph = np.load(ROOT / "04_data/prepared/validation/validation_graph.npz")
    n0, n1 = graph["graph_node_ptr"][idx : idx + 2]
    e0, e1 = graph["graph_edge_ptr"][idx : idx + 2]
    np.testing.assert_allclose(
        nodes, graph["node_features"][n0:n1], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_equal(edges, graph["edge_index"][:, e0:e1] - n0)
    np.testing.assert_allclose(
        attrs, graph["edge_features"][e0:e1], rtol=1e-5, atol=1e-6
    )


def test_recording_and_class_balance():
    m = pd.DataFrame({"split_group_id": ["a", "a", "a", "b", "c", "c"]})
    y = np.array([0, 0, 0, 0, 1, 1])
    w = recording_weights(m, y)
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
    assert np.isclose(w[:3].sum(), w[3])


def test_ingestion_adapter_renames_source_fields(tmp_path):
    folder = ROOT / "04_data/raw_examples/validation"
    row = pd.read_csv(folder / "case_index.csv").iloc[0]
    raw = pd.read_csv(folder / f"{row.sample_id}.csv").rename(
        columns={"query_type": "vendor_dns_type"}
    )
    source = tmp_path / "vendor.csv"
    target = tmp_path / "canonical.csv"
    raw.to_csv(source, index=False)
    ingest(source, target, {"vendor_dns_type": "query_type"})
    result = pd.read_csv(target)
    assert (
        len(result) == len(raw)
        and "query_type" in result
        and "vendor_dns_type" not in result
    )
