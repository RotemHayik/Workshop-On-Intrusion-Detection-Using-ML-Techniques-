import json
import pandas as pd
import pytest
from dnsids.cascade import route, parse_response, prompt_for, FEATURES


def test_budget_and_label_blind_routing():
    f = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "split_group_id": ["g1", "g1", "g2", "g3"],
            "rf_score": [0.99, 0.5, 0.99, 0.1],
            "rf_prediction": [1, 1, 1, 0],
            "if_prediction": [1, 0, 0, 0],
            "bilstm_prediction": [0, 0, 1, 0],
            "sage_prediction": [0, 1, 0, 0],
            "label": [0, 0, 0, 0],
        }
    )
    policy = {"llm_budget_per_recording": 1, "llm_budget_per_split": 1}
    a = route(f, policy)
    b = route(f.assign(label=1), policy)
    assert a.stage_two.tolist() == [False, True, True, False]
    assert a.cascade_prediction.tolist() == [1, 1, 1, 0]
    assert a.llm_requested.sum() == 1 and a.llm_requested.equals(b.llm_requested)


def test_invalid_arbitration_never_becomes_a_label():
    with pytest.raises(ValueError):
        parse_response('{"label":"uncertain"}')
    with pytest.raises(ValueError):
        parse_response('{"label":"malicious","evidence":"","limitation":"x"}')
    assert (
        parse_response(
            '{"label":"benign","evidence":"Short names","limitation":"No payload"}'
        )["label"]
        == "benign"
    )


def test_prompt_excludes_labels_and_source_identifiers():
    row = pd.Series(
        {
            **{f: 0.1 for f in FEATURES},
            **{k + "_prediction": 0 for k in ["rf", "if", "bilstm", "sage"]},
            "rf_score": 0.1,
            "label": 1,
            "source_name": "SECRET_SOURCE",
            "tool_label": "SECRET_TOOL",
            "sample_id": "SECRET_ID",
        }
    )
    prompt = json.dumps(prompt_for(row, {}))
    assert "SECRET" not in prompt and '"label"' not in prompt
