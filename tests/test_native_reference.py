from pathlib import Path

import numpy as np
import pytest
import torch

from neural_fx.native import _read_trace, native_validate, validate_plugin
from neural_fx.reference import pair_features, train_reference, infer_reference
from neural_fx.models import AudioModel
from neural_fx.utils import read_json


def test_reference_features_require_aligned_excited_audio():
    x = np.linspace(-0.3, 0.3, 512, dtype=np.float32)
    result = pair_features(x, np.tanh(x * 3))
    assert result.shape == (22,) and np.isfinite(result).all()
    with pytest.raises(ValueError, match="silent"):
        pair_features(np.zeros(512), np.zeros(512))
    with pytest.raises(ValueError, match="aligned"):
        pair_features(x, x[:-1])


def test_reference_training_and_source_disjoint_transfer(tmp_path):
    config = read_json("configs/reference_transfer.json")
    config["reference"]["epochs"] = 3
    config["data"]["samples"] = 512
    config["data"]["sources"] = {"train": 3, "valid": 2, "test": 2}
    root = tmp_path / "reference"
    result = train_reference(config, root)
    assert result["evaluation"]["cross_content_trials"] == 12
    assert np.isfinite(result["evaluation"]["test_control_mse"])
    trials = read_json(root / "evaluation/trials.json")
    assert all(row["reference_source"] != row["target_source"] for row in trials)
    path = root / "dataset/pairs/test_source_0000_preset_0"
    preset = infer_reference(root / "reference.pt", path / "dry.wav", path / "wet.wav", root / "preset.json")
    assert 0 <= preset["drive"] <= 1 and 0 <= preset["mix"] <= 1
    assert len(preset["latent"]) == 8
    with pytest.raises(FileExistsError):
        train_reference(config, root)


def test_native_trace_parser_rejects_truncation(tmp_path):
    path = tmp_path / "truncated.trace"
    path.write_bytes(b"NFX")
    with pytest.raises(ValueError, match="Truncated"):
        _read_trace(path)


@pytest.mark.skipif(not Path("build/native/Release/nafx-render.exe").exists(), reason="Build native runtime first")
@pytest.mark.parametrize("name", ["conditional_tcn", "gru", "hybrid"])
def test_native_state_audio_parity_and_host_buffer_adapter(tmp_path, name):
    pytest.importorskip("onnxscript")
    config = read_json("configs/smoke.json")
    config["model"].update({"name": name, "channels": 3, "dilations": [1, 2], "fir_taps": 9})
    model = AudioModel(**config["model"])
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_spec": config["model"], "config": config, "state_dict": model.state_dict()}, checkpoint)
    result = native_validate(checkpoint, "build/native/Release/nafx-render.exe", tmp_path / "validation", [16])
    assert result["parity_passed"]
    assert result["blocks"][0]["reset_max_error"] == 0
    assert result["blocks"][0]["benchmark"]["application_new_calls"] == 0


@pytest.mark.skipif(not Path("build/vst3-validator/bin/Release/validator.exe").exists(), reason="Build official validator first")
def test_official_vst3_conformity(tmp_path):
    plugin = Path("build/native/plugin/NeuralAudioFX_artefacts/Release/VST3/NeuralAudioFX.vst3")
    if not plugin.exists():
        pytest.skip("Build VST3 first")
    result = validate_plugin(plugin, "build/vst3-validator/bin/Release/validator.exe", tmp_path / "validator")
    assert result["passed"] and result["cases"][0]["tests_passed"] > 0
    assert not result["production_ready"]
