import numpy as np
import torch
from dnsids.metrics import choose_threshold, metrics
from dnsids.models import BiLSTM, GraphSAGE
from dnsids.data import graph_collate


def test_threshold_respects_ties_and_fpr():
    y = np.array([0] * 100 + [1] * 20)
    s = np.array([0.1] * 99 + [0.8] + [0.8] * 10 + [0.9] * 10)
    threshold = choose_threshold(y, s)
    assert metrics(y, s, threshold)["fpr"] <= 0.01
    assert metrics(y, s, threshold)["recall"] == 1


def test_malicious_only_metrics_are_not_invented():
    m = metrics([1, 1], [0.2, 0.9], 0.5)
    assert m["recall"] == 0.5 and m["fpr"] is None and m["auroc"] is None


def test_benign_source_reports_false_positive_rate():
    m = metrics([0, 0], [0.2, 0.9], 0.5)
    assert m["fpr"] == 0.5 and m["recall"] is None and m["auroc"] is None


def test_bilstm_ignores_padding_in_both_directions():
    torch.manual_seed(0)
    m = BiLSTM().eval()
    x = torch.randn(2, 128, 15)
    q = torch.full((2, 128), 2, dtype=torch.long)
    r = q.clone()
    length = torch.tensor([32, 128])
    with torch.no_grad():
        before = m(x, q, r, length)
        x[0, 32:] = 500
        q[0, 32:] = 9
        r[0, 32:] = 8
        after = m(x, q, r, length)
    assert torch.allclose(before, after, atol=1e-6)


def test_graph_batch_has_no_cross_graph_information():
    torch.manual_seed(0)
    m = GraphSAGE().eval()
    a = (
        torch.randn(3, 13),
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        torch.tensor(0.0),
    )
    b = (
        torch.randn(5, 13),
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        torch.tensor(1.0),
    )
    with torch.no_grad():
        alone = m(*graph_collate([a])[:-1])[0]
        together = m(*graph_collate([a, b])[:-1])[0]
    assert torch.allclose(alone, together, atol=1e-6)
