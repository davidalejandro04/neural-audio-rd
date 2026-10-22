from copy import deepcopy

import numpy as np
import pytest
import torch

from neural_fx.data import generate
from neural_fx.experiment import load_checkpoint
from neural_fx.pipeline import run
from neural_fx.study import analyze_study, matched_baseline, paired_effect, parameter_count, plan_study
from neural_fx.utils import canonical_hash, read_json, sha256, write_json


def test_parameter_matching_and_planner_preserve_input_and_rng(tmp_path):
    config = read_json("configs/research.json")
    original = deepcopy(config)
    torch.manual_seed(41)
    state = torch.get_rng_state().clone()
    baseline = matched_baseline(config["model"])
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert baseline["channels"] == 19
    assert parameter_count(baseline) == 6138
    result = plan_study(config, tmp_path / "study")
    assert config == original
    assert result["training_runs"] == 24
    plan = read_json(tmp_path / "study/plan.json")
    assert len(plan["entries"]) == 6
    for entry in plan["entries"]:
        saved = read_json(tmp_path / "study" / entry["config"])
        assert saved["data_seed"] == 8181
        assert canonical_hash(saved) == entry["config_sha256"]
        assert saved["model_variants"]["conditional_tcn"]["channels"] == 19
    with pytest.raises(FileExistsError):
        plan_study(config, tmp_path / "study")


def test_fixed_data_seed_is_independent_of_training_seed(tmp_path):
    config = read_json("configs/smoke.json")
    config["data"]["sources"] = {"train": 1, "valid": 1, "test": 1}
    config["data"]["samples"] = 512
    config["data_seed"] = 8181
    a = generate(config, tmp_path / "a")
    config["seed"] = 2027
    b = generate(config, tmp_path / "b")
    assert a == b
    config["data_seed"] = 8182
    c = generate(config, tmp_path / "c")
    assert a["manifest_sha256"] != c["manifest_sha256"]


def rows(scale=1.0):
    result = []
    for source in range(4):
        for preset in range(2):
            result.append({"source_id": str(source), "pair_id": f"{source}_{preset}",
                           "regime": "seen_controls" if preset == 0 else "held_out_controls",
                           "system": "model", "bypass": False,
                           "esr": (source + 1) * 0.01 * scale,
                           "mrstft": (source + 1) * 0.1 * scale})
    return result


def test_paired_bootstrap_known_improvement_and_degradation():
    criteria = {"bootstrap_seed": 1, "bootstrap_draws": 200, "relative_improvement": 0.1}
    pairs = [(rows(), rows(0.5))] * 3
    result = paired_effect(pairs, criteria)
    for value in result["metrics"].values():
        assert value["relative_improvement"] == pytest.approx(0.5)
        assert value["absolute_improvement_95_ci"][0] > 0
        assert value["improvement_supported"]
    assert result == paired_effect(pairs, criteria)
    worse = paired_effect([(rows(), rows(2))] * 3, criteria, "held_out_controls")
    assert not worse["metrics"]["esr"]["improvement_supported"]
    with pytest.raises(ValueError, match="differ"):
        paired_effect([(rows(), rows()[:-1])], criteria)


def test_pipeline_applies_model_variant_override(tmp_path):
    config = read_json("configs/smoke.json")
    config["training"]["epochs"] = 1
    config["data"]["sources"] = {"train": 1, "valid": 1, "test": 1}
    config["data"]["samples"] = 512
    config["evaluation"] = {"block_sizes": [64], "audio_examples": 0}
    config["suite"] = ["conditional_tcn"]
    config["model_variants"] = {"conditional_tcn": {"channels": 3}}
    result = run(config, tmp_path / "run", suite=True)
    assert result["state"] == "completed"
    model, saved = load_checkpoint(tmp_path / "run/models/conditional_tcn/best.pt")
    assert model.channels == 3 and saved["model_spec"]["channels"] == 3


def make_study_evidence(tmp_path):
    plan_study(read_json("configs/research.json"), tmp_path / "study")
    root = tmp_path / "study"
    plan = read_json(root / "plan.json")
    # Small deterministic fixtures test decisions, not research acceptance data.
    plan["criteria"].update({"minimum_test_sources": 4, "bootstrap_draws": 100})
    write_json(root / "plan.json", plan)
    for entry in plan["entries"]:
        folder = root / entry["run"]
        write_json(folder / "preregistered_config.json", read_json(root / entry["config"]))
        write_json(folder / "status.json", {"state": "completed"})
        for name, scale, count in (("conditional_tcn", 1.0, 6138), ("hybrid", 0.5, 6011)):
            model = folder / "models" / name
            model.mkdir(parents=True)
            (model / "best.pt").write_bytes(b"test fixture, never loaded")
            write_json(model / "evaluation/metrics.json", {"parameters": count, "checkpoint_sha256": sha256(model / "best.pt"),
                                                          "manifest_sha256": {"test": "fixed-test-sources"}})
            write_json(model / "evaluation/per_pair.json", rows(scale))
            write_json(model / "gates.json", {"checks": {key: {"passed": True} for key in ("runtime_screening", "streaming", "quality_esr", "beats_identity")}})
    return root


def test_study_analysis_requires_complete_evidence_and_runtime(tmp_path):
    root = make_study_evidence(tmp_path)
    result = analyze_study(root / "plan.json", root / "analysis.json")
    assert result["p3_accepted"]
    assert not result["production_ready"]
    plan = read_json(root / "plan.json")
    folder = root / plan["entries"][0]["run"] / "models/hybrid"
    gates = read_json(folder / "gates.json")
    gates["checks"]["runtime_screening"]["passed"] = False
    write_json(folder / "gates.json", gates)
    result = analyze_study(root / "plan.json", root / "analysis.json")
    assert not result["p3_accepted"]
    assert result["effects"]["distortion"]["quality_hypothesis_supported"]
    assert not result["effects"]["distortion"]["runtime_screening"]
    write_json(root / plan["entries"][0]["run"] / "status.json", {"state": "failed"})
    result = analyze_study(root / "plan.json", root / "analysis.json")
    assert result["state"] == "incomplete" and not result["p3_accepted"]


def test_study_rejects_mismatched_snapshots(tmp_path):
    root = make_study_evidence(tmp_path)
    plan = read_json(root / "plan.json")
    path = root / plan["entries"][0]["run"] / "models/hybrid/evaluation/metrics.json"
    metrics = read_json(path)
    metrics["manifest_sha256"] = {"test": "different"}
    write_json(path, metrics)
    with pytest.raises(ValueError, match="different datasets"):
        analyze_study(root / "plan.json", root / "analysis.json")
