import json

import pytest

from sage.eval.calc import calc_metrics
from sage.train.build_anchor_bank import extract_sid_codes


def test_extract_sid_codes_accepts_collision_token():
    index = {"x": ["<|a_1|>", "<|b_2|>", "<|c_3|>", "<|d_256|>", "<|e_4|>"]}
    codes = extract_sid_codes(index, ["x"])
    assert codes.tolist() == [[0, 1, 2, 255]]


def test_metric_calculation(tmp_path):
    target = "<|a_1|><|b_2|><|c_3|><|d_4|>"
    other = "<|a_2|><|b_2|><|c_3|><|d_4|>"
    index_path = tmp_path / "Beauty.index.json"
    result_path = tmp_path / "predictions.json"
    metrics_path = tmp_path / "metrics.json"
    index_path.write_text(
        json.dumps(
            {
                "target": ["<|a_1|>", "<|b_2|>", "<|c_3|>", "<|d_4|>"],
                "other": ["<|a_2|>", "<|b_2|>", "<|c_3|>", "<|d_4|>"],
            }
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps([{"output": target, "predict": [other, target, other]}]),
        encoding="utf-8",
    )

    result = calc_metrics([str(result_path)], str(index_path), save_path=str(metrics_path))[
        str(result_path)
    ]
    assert result["HR"] == [0.0, 1.0]
    assert result["NDCG"][0] == 0.0
    assert result["NDCG"][1] == pytest.approx(1.0 / 1.584962500721156)
    assert result["oov_predictions"] == 0
