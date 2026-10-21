from pathlib import Path

import pytest
import torch

from neural_fx.data import generate
from neural_fx.experiment import evaluate, load_checkpoint, train
from neural_fx.pipeline import assess
from neural_fx.pipeline import run
from neural_fx.utils import read_json


def test_training_selects_validation_checkpoint_and_evaluates(tmp_path):
    config = read_json("configs/smoke.json")
    config["training"]["epochs"] = 2
    config["data"]["sources"] = {"train": 2, "valid": 1, "test": 1}
    config["data"]["samples"] = 512
    config["evaluation"]["audio_examples"] = 0
    data = tmp_path / "data"
    generate(config, data)
    first, second = tmp_path / "first", tmp_path / "second"
    train(config, data, first)
    train(config, data, second)
    a, _ = load_checkpoint(first / "best.pt")
    b, _ = load_checkpoint(second / "best.pt")
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    scores = evaluate(first / "best.pt", data, first / "evaluation")
    assert scores["summary"]["analytic_oracle"]["active_source_median"]["esr"] < 1e-12
    assert scores["max_stream_error"] < 1e-5
    assert scores["independent_test_sources"] == 1
    with pytest.raises(FileExistsError):
        train(config, data, first)


def test_gate_rejects_bad_quality_and_slow_runtime():
    config = read_json("configs/smoke.json")
    scores = {"summary": {"model": {"active_source_median": {"esr": 0.6}},
                           "identity": {"active_source_median": {"esr": 0.7}}}, "max_stream_error": 0}
    result = assess(config, scores, {"blocks": [{"p99_budget_fraction": 2.0}]})
    assert not result["passed"]
    assert not result["checks"]["quality_esr"]["passed"]
    assert not result["checks"]["runtime_screening"]["passed"]
    assert not result["production_ready"]


def test_generation_failure_is_recorded(tmp_path):
    config = read_json("configs/smoke.json")
    config["data"]["samples"] = 8
    with pytest.raises(ValueError, match="samples >= 256"):
        run(config, tmp_path / "failed")
    status = read_json(tmp_path / "failed/status.json")
    assert status["state"] == "failed" and status["stage"] == "generate"
