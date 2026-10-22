import numpy as np
import pytest
import torch
from scipy.signal import convolve

from neural_fx.dsp import distortion, impulse_response, metrics, target
from neural_fx.models import AudioModel, stream

NAMES = ("conv1d", "tcn", "conditional_tcn", "gru", "hybrid_no_gru", "hybrid_no_fir", "hybrid")
torch.set_num_threads(1)


def test_analytic_targets():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    np.testing.assert_array_equal(distortion(x, 0.5, 0), x)
    np.testing.assert_allclose(distortion(x, 0.5, 1), -distortion(-x, 0.5, 1))
    for effect in ("cabinet", "reverb"):
        ir = impulse_response(48000, effect)
        np.testing.assert_allclose(target(x, [1, 1], effect, 48000), convolve(x, ir)[:len(x)], atol=1e-6)
        impulse = np.zeros(len(ir), dtype=np.float32)
        impulse[0] = 1
        np.testing.assert_allclose(target(impulse, [1, 1], effect, 48000), ir, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 3])
def test_shapes_bypass_and_numerical_stability(name, batch):
    torch.manual_seed(1)
    model = AudioModel(name).eval()
    signals = [torch.zeros(batch, 1, 535), torch.ones(batch, 1, 535), torch.randn(batch, 1, 535)]
    for x in signals:
        bypass = torch.tensor([[0.7, 0.0]]).expand(batch, -1)
        torch.testing.assert_close(model(x, bypass), x, rtol=0, atol=0)
        controls = torch.tensor([[0.7, 1.0]]).expand(batch, -1)
        y = model(x, controls)
        assert y.shape == x.shape and torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_streaming_reset_and_causality(name):
    torch.manual_seed(1337)
    model = AudioModel(name).eval()
    x = torch.randn(1, 1, 1103) * 0.2
    controls = torch.tensor([[0.4, 1.0]])
    with torch.inference_mode():
        reference = model(x, controls)
        for sizes in ((64,), (128,), (256,), (512,), (7, 64, 31, 256)):
            actual = stream(model, x, controls, sizes)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        # A fresh stream resets state independently of preceding calls.
        torch.testing.assert_close(stream(model, x, controls), stream(model, x, controls))
        modified = x.clone()
        modified[..., 500:] = torch.randn_like(modified[..., 500:])
        torch.testing.assert_close(model(modified, controls)[..., :500], reference[..., :500])


def test_metrics_perfect_and_silence():
    x = torch.randn(1, 1, 1024)
    result = metrics(x, x)
    assert result["esr"] == 0 and result["mrstft"] == 0 and result["lsd_db"] == 0
    assert metrics(torch.zeros_like(x), torch.zeros_like(x))["esr"] is None


@pytest.mark.parametrize("name", ["conditional_tcn", "gru", "hybrid", "hybrid_no_gru"])
def test_wet_path_v2_obeys_physical_mix_law(name):
    model = AudioModel(name, wet_only_conditioning=True).eval()
    x = torch.randn(1, 1, 535) * 0.2
    with torch.inference_mode():
        fully_wet = model(x, torch.tensor([[0.4, 1.0]]))
        for mix in (0.0, 0.3, 0.7):
            controls = torch.tensor([[0.4, mix]])
            expected = (1 - mix) * x + mix * fully_wet
            torch.testing.assert_close(model(x, controls), expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(stream(model, x, controls, (17, 64, 113)), expected, atol=1e-6, rtol=1e-5)
