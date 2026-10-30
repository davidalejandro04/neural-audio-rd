"""The new target is causal, gain-conditioned, and separate from cabinet DSP."""
import numpy as np
import torch
import pytest
from pathlib import Path

from neural_fx.dsp import target
from neural_fx.models import build_model, stream


def test_high_gain_target_causal_bypass_silence_and_drive():
    rng = np.random.default_rng(42)
    x = rng.normal(0, .025, 4096).astype(np.float32)
    assert np.array_equal(target(x, (.9, 0), "high_gain", 48000), x)
    assert not np.any(target(np.zeros_like(x), (1, 1), "high_gain", 48000))
    light = target(x, (.1, 1), "high_gain", 48000)
    heavy = target(x, (.9, 1), "high_gain", 48000)
    assert np.isfinite(heavy).all() and np.max(np.abs(heavy)) <= 1
    assert np.mean(heavy ** 2) > 3 * np.mean(light ** 2)
    changed = x.copy()
    changed[2048:] *= -1
    assert np.array_equal(target(changed, (.9, 1), "high_gain", 48000)[:2048], heavy[:2048])
    np.testing.assert_allclose(target(x, (.9, .3), "high_gain", 48000), .7*x + .3*heavy, atol=1e-7)


def test_high_gain_architecture_stream_and_mix_contract():
    model = build_model(dict(name="conditional_tcn", channels=16, dilations=[1,2,4,8,16,32], wet_only_conditioning=True, zero_preserving=True)).eval()
    x = torch.randn(1, 1, 2051) * .1
    with torch.inference_mode():
        wet = model(x, torch.tensor([[.85, 1.]]))
        mixed = model(x, torch.tensor([[.85, .3]]))
        torch.testing.assert_close(mixed, .7*x + .3*wet, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(stream(model, x, torch.tensor([[.85, .3]]), (17,64,113,512)), mixed, atol=1e-6, rtol=1e-5)
        for drive in (0., .5, 1.):
            silence = stream(model, torch.zeros_like(x), torch.tensor([[drive, 1.]]), (17, 64, 113))
            assert torch.count_nonzero(silence) == 0


@pytest.mark.skipif(not Path("build/native/Release/nafx-render.exe").exists(), reason="Build native runtime first")
def test_native_rejects_incompatible_model_control_metadata(tmp_path):
    import subprocess
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    from neural_fx.runtime import export_onnx
    from neural_fx.utils import read_json, seed_everything

    seed_everything(1337)
    config = read_json("experiments/high-gain.json")
    model = build_model(config["model"])
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_spec": config["model"], "config": config, "state_dict": model.state_dict()}, checkpoint)
    path = tmp_path / "model.onnx"
    exported = export_onnx(checkpoint, path, 128)
    assert exported["passed"] and exported["effect"] == "high_gain"
    graph = onnx.load(path)
    metadata = {p.key: p.value for p in graph.metadata_props}
    assert metadata["neural_fx_control_order"] == "drive,mix"
    for key, value, message in (("neural_fx_control_order", "mix,drive", "Unsupported control order"),
                                ("neural_fx_mix_contract", "wet_only", "Unsupported mix contract")):
        onnx.helper.set_model_props(graph, {**metadata, key: value})
        onnx.save(graph, path)
        result = subprocess.run(["build/native/Release/nafx-render.exe", "benchmark", "--model", str(path),
                                 "--output", str(tmp_path / "timing.json"), "--iterations", "20"], capture_output=True, text=True)
        assert result.returncode != 0 and message in result.stderr
