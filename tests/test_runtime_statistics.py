import pytest

from neural_fx.runtime import _block_statistics, benchmark_onnx


def test_deadline_statistics_count_misses_and_units():
    result = _block_statistics([0.1, 0.2, 2.0], 48, 48000)
    assert result["deadline_ms"] == 1
    assert result["deadline_misses"] == 1
    assert result["mean_ms"] == pytest.approx(2.3 / 3)
    assert result["p99_budget_fraction"] == result["p99_ms"]


def test_benchmark_rejects_too_few_iterations(tmp_path):
    pytest.importorskip("onnxruntime")
    with pytest.raises(ValueError, match="20"):
        benchmark_onnx(tmp_path / "unused.onnx", tmp_path / "result.json", 0)
