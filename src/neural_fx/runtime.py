"""Stateful CPU measurements and a fixed-block ONNX interface."""
import time
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiment import load_checkpoint
from .utils import canonical_hash, environment, read_json, seed_everything, sha256, write_json


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


def _block_statistics(elapsed, block_size, sample_rate):
    deadline = block_size / sample_rate * 1000
    return {"block_size": block_size, "deadline_ms": deadline,
            "mean_ms": float(np.mean(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)),
            "p99_ms": float(np.percentile(elapsed, 99)), "worst_ms": float(np.max(elapsed)),
            "p99_budget_fraction": float(np.percentile(elapsed, 99)) / deadline,
            "deadline_misses": int(np.sum(np.asarray(elapsed) > deadline)),
            "realtime_factor": float(np.mean(elapsed)) / deadline}


def benchmark_onnx(model_path, output, iterations=2000, checkpoint=None):
    """Paired fixed-block CPU screen; never upgrades a research/release gate."""
    import onnxruntime as ort

    if iterations < 20:
        raise ValueError("Use at least 20 measured iterations")
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    golden_path = model_path.with_suffix(".golden.npz")
    if Path(output).resolve() in {path.resolve() for path in (model_path, metadata_path, golden_path)}:
        raise ValueError("Benchmark output must not overwrite model or export evidence")
    metadata = read_json(metadata_path)
    if "hidden" in metadata["inputs"]:
        raise ValueError("benchmark-onnx currently screens finite-context exports; use native validation for recurrent state")
    if not metadata["passed"] or metadata["onnx_sha256"] != sha256(model_path):
        raise ValueError("Export parity was not accepted or ONNX checksum changed")
    rate, size = metadata["sample_rate"], metadata["block_size"]
    if rate <= 0 or size <= 0:
        raise ValueError("Export metadata needs positive sample rate and block size")
    seed_everything(1337)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    history_shape = tuple(metadata["inputs"]["history"])
    # Replay the complete saved sequence, carrying state across every block.
    maximum = 0.0
    history = np.zeros(history_shape, dtype=np.float32)
    with np.load(golden_path, allow_pickle=False) as fixtures:
        lengths = [len(fixtures[key]) for key in ("audio", "controls", "expected", "next_history")]
        if min(lengths) == 0 or len(set(lengths)) != 1:
            raise ValueError("Golden fixture sequences must be nonempty and equally long")
        for audio, controls, expected, expected_history in zip(*(fixtures[key] for key in ("audio", "controls", "expected", "next_history"))):
            wet, history = session.run(["wet", "next_history"], {"audio": audio, "controls": controls, "history": history})
            if not np.isfinite(wet).all() or not np.isfinite(history).all():
                raise ValueError("Non-finite golden sequence output")
            maximum = max(maximum, float(np.max(np.abs(wet - expected))), float(np.max(np.abs(history - expected_history))))
    if maximum > metadata["tolerance"]:
        raise ValueError(f"Golden-sequence parity failed: {maximum}")
    rng = np.random.default_rng(1337)
    audio_pool = rng.uniform(-0.2, 0.2, (16, 1, 1, size)).astype(np.float32)
    controls_pool = rng.uniform(0.1, 1.0, (16, 1, 2)).astype(np.float32)
    controls_pool[:, :, 1] = 1.0  # Active effect, no bypass shortcuts.
    feed = {"audio": audio_pool[0], "controls": controls_pool[0], "history": np.zeros(history_shape, dtype=np.float32)}
    for index in range(20):
        feed["audio"], feed["controls"] = audio_pool[index % 16], controls_pool[index % 16]
        _, feed["history"] = session.run(["wet", "next_history"], feed)
    elapsed = np.empty(iterations)
    wall_start = time.perf_counter()
    for index in range(iterations):
        feed["audio"], feed["controls"] = audio_pool[index % 16], controls_pool[index % 16]
        start = time.perf_counter_ns()
        wet, feed["history"] = session.run(["wet", "next_history"], feed)
        elapsed[index] = (time.perf_counter_ns() - start) / 1e6
        if not np.isfinite(wet).all() or not np.isfinite(feed["history"]).all():
            raise ValueError("Non-finite benchmark output")
    wall_seconds = time.perf_counter() - wall_start
    onnx_stats = _block_statistics(elapsed, size, rate)
    report = {"runtime": "ONNX Runtime CPU via Python; unpaced fixed-block screening",
              "onnx_sha256": sha256(model_path), "checkpoint_sha256": metadata["checkpoint_sha256"],
              "golden_sha256": sha256(golden_path), "golden_max_error": maximum,
              "providers": session.get_providers(), "onnxruntime": ort.__version__,
              "threads": 1, "warmup_blocks": 20, "iterations": iterations,
              "sample_rate": rate, "simulated_audio_seconds": iterations * size / rate,
              "wall_seconds": wall_seconds, "onnx": onnx_stats,
              "screening_criteria": {"max_p99_budget_fraction": 0.7, "maximum_deadline_misses": 0},
              "environment": environment(), "production_ready": False,
              "limitations": "Includes Python dispatch and output allocations; excludes input selection and finite checks from timed intervals. Unpaced execution is not a sustained native audio-callback test. Only the exported block size is measured."}
    if checkpoint is not None:
        if sha256(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError("Comparison checkpoint differs from the exported model")
        model, saved = load_checkpoint(checkpoint)
        if model.has_gru:
            raise ValueError("This ONNX benchmark adapter does not support recurrent exports")
        if saved["config"]["data"]["sample_rate"] != rate:
            raise ValueError("Comparison checkpoint sample rate differs from export")
        x_pool = [torch.from_numpy(x) for x in audio_pool]
        c_pool = [torch.from_numpy(c) for c in controls_pool]
        torch_elapsed = np.empty(iterations)
        with torch.inference_mode():
            history, hidden = model.initial_state()
            for index in range(20):
                _, history, hidden = model.process(x_pool[index % 16], c_pool[index % 16], history, hidden)
            for index in range(iterations):
                x, c = x_pool[index % 16], c_pool[index % 16]
                start = time.perf_counter_ns()
                wet, history, hidden = model.process(x, c, history, hidden)
                torch_elapsed[index] = (time.perf_counter_ns() - start) / 1e6
                if not torch.isfinite(wet).all() or not torch.isfinite(history).all():
                    raise ValueError("Non-finite PyTorch benchmark output")
        report["pytorch"] = _block_statistics(torch_elapsed, size, rate)
        report["p99_speedup"] = report["pytorch"]["p99_ms"] / onnx_stats["p99_ms"]
    report["screening_passed"] = onnx_stats["p99_budget_fraction"] <= 0.7 and onnx_stats["deadline_misses"] == 0
    write_json(output, report)
    return report


def benchmark_study(plan_path, output, iterations=2000):
    plan_path, output = Path(plan_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Benchmark directory already exists: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path)
    rows = []
    write_json(output / "status.json", {"state": "running", "completed_models": []})
    try:
        for entry in plan["entries"]:
            root = plan_path.parent / entry["run"]
            if read_json(root / "status.json")["state"] != "completed":
                raise ValueError(f"Study run is incomplete: {root}")
            if canonical_hash(read_json(root / "preregistered_config.json")) != entry["config_sha256"]:
                raise ValueError(f"Study config differs from plan: {root}")
            for name in ("conditional_tcn", "hybrid_no_gru"):
                folder = root / "models" / name
                report_name = f"{entry['effect']}-{entry['seed']}-{name}.json"
                report = benchmark_onnx(folder / "model.onnx", output / report_name, iterations, folder / "best.pt")
                rows.append({"effect": entry["effect"], "seed": entry["seed"], "model": name,
                             "block_size": report["onnx"]["block_size"], "onnx_p99_ms": report["onnx"]["p99_ms"],
                             "pytorch_p99_ms": report["pytorch"]["p99_ms"], "p99_speedup": report["p99_speedup"],
                             "onnx_p99_budget_fraction": report["onnx"]["p99_budget_fraction"],
                             "deadline_misses": report["onnx"]["deadline_misses"],
                             "screening_passed": report["screening_passed"]})
                write_json(output / "status.json", {"state": "running", "completed_models": rows})
                print(f"{entry['effect']} seed={entry['seed']} {name}: ONNX p99={report['onnx']['p99_ms']:.4f} ms, screening={'PASS' if report['screening_passed'] else 'FAIL'}", flush=True)
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error), "completed_models": rows})
        raise
    result = {"state": "completed", "plan_sha256": sha256(plan_path), "comparison": rows,
              "scope": "12 existing nonrecurrent exports; no retraining or update to P3 acceptance",
              "remaining": ["persistent GRU export/runtime", "native allocation audit", "64/256/512 block tests", "sustained callback measurement", "real capture validation"],
              "production_ready": False}
    write_json(output / "summary.json", result)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "status.json", {"state": "completed", "completed_models": rows})
    return result


class ExportBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, controls, history):
        # Initial export supports finite-context architectures only; no fake GRU reset.
        hidden = audio.new_zeros(1, 1, self.model.channels)
        y, history, _ = self.model.process(audio, controls, history, hidden)
        return y, history


class ExportRecurrentBlock(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, controls, history, hidden):
        wet, next_history, next_hidden = self.model.process(audio, controls, history, hidden)
        # Keep the uniform history port even for a pure GRU whose raw history is otherwise redundant.
        return wet, next_history + history * 0.0, next_hidden


def export_onnx(checkpoint, output, block_size=128):
    import onnx
    import onnxruntime as ort

    seed_everything(1337)
    model, saved = load_checkpoint(checkpoint)
    if block_size < 1:
        raise ValueError("Positive block size required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = (ExportRecurrentBlock(model) if model.has_gru else ExportBlock(model)).eval()
    controls = torch.tensor([[0.5, 1.0]])
    history, hidden = model.initial_state()
    x = torch.randn(1, 1, block_size) * 0.1
    input_names = ["audio", "controls", "history"] + (["hidden"] if model.has_gru else [])
    output_names = ["wet", "next_history"] + (["next_hidden"] if model.has_gru else [])
    args = (x, controls, history, hidden) if model.has_gru else (x, controls, history)
    torch.onnx.export(wrapper, args, str(output), dynamo=True,
                      input_names=input_names, output_names=output_names, opset_version=18,
                      external_data=False, verbose=False)
    graph = onnx.load(output)
    onnx.helper.set_model_props(graph, {"neural_fx_sample_rate": str(saved["config"]["data"]["sample_rate"]),
                                       "neural_fx_block_size": str(block_size), "neural_fx_state_version": "1"})
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    ort_history = history.numpy()
    ort_hidden = hidden.numpy()
    maximum = 0.0
    golden, golden_inputs, golden_controls, golden_history, golden_hidden = [], [], [], [], []
    with torch.inference_mode():
        for step in range(12):
            x = torch.randn(1, 1, block_size) * 0.1
            if step == 0:
                x.zero_()
                x[..., 0] = 1
            controls = torch.tensor([[step / 12, 0.0 if step == 4 else 1.0]])
            feed = {"audio": x.numpy(), "controls": controls.numpy(), "history": ort_history}
            if model.has_gru:
                expected, history, hidden = wrapper(x, controls, history, hidden)
                feed["hidden"] = ort_hidden
                actual, ort_history, ort_hidden = session.run(output_names, feed)
                maximum = max(maximum, float(np.max(np.abs(ort_hidden - hidden.numpy()))))
            else:
                expected, history = wrapper(x, controls, history)
                actual, ort_history = session.run(output_names, feed)
            maximum = max(maximum, float(np.max(np.abs(actual - expected.numpy()))),
                          float(np.max(np.abs(ort_history - history.numpy()))))
            golden.append(expected.numpy())
            golden_inputs.append(x.numpy())
            golden_controls.append(controls.numpy())
            golden_history.append(history.numpy())
            golden_hidden.append(hidden.numpy())
    tolerance = saved["config"]["gates"]["export_max_error"]
    report = {"passed": maximum <= tolerance, "max_error": maximum, "tolerance": tolerance,
              "checkpoint_sha256": sha256(checkpoint), "onnx_sha256": sha256(output),
              "block_size": block_size, "sample_rate": saved["config"]["data"]["sample_rate"],
              "inputs": {"audio": [1, 1, block_size], "controls": [1, 2], "history": [1, 1, model.history_samples]},
              "control_order": ["drive", "mix"], "exporter": "torch.onnx dynamo=True",
              "state_policy": "Carry next_history into next call; zero on stream reset"}
    if model.has_gru:
        report["inputs"]["hidden"] = list(hidden.shape)
        report["state_policy"] = "Carry next_history AND next_hidden into next call; zero both on reset"
    write_json(output.with_suffix(".json"), report)
    np.savez(output.with_suffix(".golden.npz"), audio=np.stack(golden_inputs),
             controls=np.stack(golden_controls), expected=np.stack(golden),
             next_history=np.stack(golden_history), next_hidden=np.stack(golden_hidden))
    if not report["passed"]:
        raise ValueError(f"ONNX parity failed: {maximum}")
    return report
