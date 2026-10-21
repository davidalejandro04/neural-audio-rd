"""Stateful CPU measurements and a fixed-block ONNX interface."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiment import load_checkpoint
from .utils import environment, seed_everything, sha256, write_json


@torch.inference_mode()
def benchmark(checkpoint, output, block_sizes=(64, 128, 256, 512), iterations=200):
    if iterations < 20 or not block_sizes or any(b < 1 for b in block_sizes):
        raise ValueError("Benchmark needs positive block sizes and at least 20 iterations")
    seed_everything(1337)
    model, saved = load_checkpoint(checkpoint)
    rate = saved["config"]["data"]["sample_rate"]
    timings = []
    for size in block_sizes:
        x = torch.randn(1, 1, size) * 0.1
        controls = torch.tensor([[0.5, 1.0]])
        history, hidden = model.initial_state()
        for _ in range(20):
            _, history, hidden = model.process(x, controls, history, hidden)
        elapsed = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            _, history, hidden = model.process(x, controls, history, hidden)
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        deadline = size / rate * 1000
        timings.append({"block_size": size, "deadline_ms": deadline, "mean_ms": float(np.mean(elapsed)),
                        "p95_ms": float(np.percentile(elapsed, 95)), "p99_ms": float(np.percentile(elapsed, 99)),
                        "worst_ms": max(elapsed), "p99_budget_fraction": float(np.percentile(elapsed, 99)) / deadline,
                        "deadline_misses": sum(value > deadline for value in elapsed),
                        "realtime_factor": float(np.mean(elapsed)) / deadline})
    report = {"runtime": "PyTorch CPU; engineering screening, not an audio-callback guarantee",
              "checkpoint_sha256": sha256(checkpoint), "environment": environment(),
              "iterations": iterations, "sample_rate": rate, "algorithmic_lookahead_samples": 0,
              "blocks": timings}
    write_json(output, report)
    return report


class ExportBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, controls, history):
        # Initial export supports finite-context architectures only; no fake GRU reset.
        hidden = audio.new_zeros(1, 1, self.model.channels)
        y, history, _ = self.model.process(audio, controls, history, hidden)
        return y, history


def export_onnx(checkpoint, output, block_size=128):
    import onnx
    import onnxruntime as ort

    seed_everything(1337)
    model, saved = load_checkpoint(checkpoint)
    if model.has_gru:
        raise ValueError("GRU export requires an explicit hidden-state adapter; export TCN or hybrid_no_gru for now")
    if block_size < 1:
        raise ValueError("Positive block size required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportBlock(model).eval()
    controls = torch.tensor([[0.5, 1.0]])
    history, _ = model.initial_state()
    x = torch.randn(1, 1, block_size) * 0.1
    torch.onnx.export(wrapper, (x, controls, history), str(output), dynamo=True,
                      input_names=["audio", "controls", "history"],
                      output_names=["wet", "next_history"], opset_version=18,
                      external_data=False, verbose=False)
    onnx.checker.check_model(onnx.load(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    ort_history = history.numpy()
    maximum = 0.0
    golden, golden_inputs, golden_controls, golden_history = [], [], [], []
    with torch.inference_mode():
        for step in range(12):
            x = torch.randn(1, 1, block_size) * 0.1
            if step == 0:
                x.zero_()
                x[..., 0] = 1
            controls = torch.tensor([[step / 12, 0.0 if step == 4 else 1.0]])
            expected, history = wrapper(x, controls, history)
            actual, ort_history = session.run(None, {"audio": x.numpy(), "controls": controls.numpy(), "history": ort_history})
            maximum = max(maximum, float(np.max(np.abs(actual - expected.numpy()))),
                          float(np.max(np.abs(ort_history - history.numpy()))))
            golden.append(expected.numpy())
            golden_inputs.append(x.numpy())
            golden_controls.append(controls.numpy())
            golden_history.append(history.numpy())
    tolerance = saved["config"]["gates"]["export_max_error"]
    report = {"passed": maximum <= tolerance, "max_error": maximum, "tolerance": tolerance,
              "checkpoint_sha256": sha256(checkpoint), "onnx_sha256": sha256(output),
              "block_size": block_size, "sample_rate": saved["config"]["data"]["sample_rate"],
              "inputs": {"audio": [1, 1, block_size], "controls": [1, 2], "history": [1, 1, model.history_samples]},
              "control_order": ["drive", "mix"], "exporter": "torch.onnx dynamo=True",
              "state_policy": "Carry next_history into next call; zero on stream reset"}
    write_json(output.with_suffix(".json"), report)
    np.savez(output.with_suffix(".golden.npz"), audio=np.stack(golden_inputs),
             controls=np.stack(golden_controls), expected=np.stack(golden),
             next_history=np.stack(golden_history))
    if not report["passed"]:
        raise ValueError(f"ONNX parity failed: {maximum}")
    return report
