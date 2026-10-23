"""Build, exercise and audit the native runtime without shell-specific scripts."""
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from .data import read_audio
from .experiment import load_checkpoint
from .models import stream
from .runtime import export_onnx
from .utils import environment, read_json, seed_everything, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "ort": ("onnxruntime-win-x64-1.23.2", "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip", "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd"),
    "juce": ("JUCE-8.0.12", "https://codeload.github.com/juce-framework/JUCE/zip/refs/tags/8.0.12", "91ceb4b54f9cc763884f6285fcfee4372fbe72d33061a339255cee2957e34ad8"),
}
VST3_REVISION = "3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96"


def setup_dependency(name):
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise ValueError("Automatic SDK setup supports Windows x64; supply --sdk and --juce paths on other platforms")
    folder, url, digest = DEPENDENCIES[name]
    root = REPO / ".deps/native"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{folder}.zip"
    if not archive.exists():
        print(f"Downloading pinned {folder}...", flush=True)
        partial = archive.with_suffix(".download")
        urllib.request.urlretrieve(url, partial)
        if sha256(partial) != digest:
            raise ValueError(f"Downloaded SDK checksum mismatch: {folder}")
        partial.replace(archive)
    if sha256(archive) != digest:
        raise ValueError(f"Cached SDK checksum mismatch: {archive}")
    destination = root / folder
    if not destination.exists():
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if not (root / entry.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("SDK archive path escapes extraction directory")
            zipped.extractall(root)
    return destination


def _run(command, log=None, env=None):
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env, check=False)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"Command failed ({result.returncode}): {command[0]}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout


def native_build(output, sdk=None, plugin=False, juce=None):
    if not shutil.which("cmake"):
        raise ValueError("CMake must be installed and available on PATH")
    output = Path(output).resolve()
    sdk = Path(sdk).resolve() if sdk else setup_dependency("ort")
    command = ["cmake", "-S", REPO / "deploy/native", "-B", output, f"-DORT_ROOT={sdk}"]
    if os.name == "nt":
        command += ["-A", "x64"]
    if plugin:
        juce = Path(juce).resolve() if juce else setup_dependency("juce")
        command += [f"-DJUCE_ROOT={juce}"]
    print("Configuring native build...", flush=True)
    _run(command, output / "configure.log")
    targets = ["nafx-render"] + (["NeuralAudioFX_VST3", "NeuralAudioFX_Standalone", "nafx-plugin-check", "nafx-ui-check"] if plugin else [])
    print("Building " + ", ".join(targets) + "...", flush=True)
    _run(["cmake", "--build", output, "--config", "Release", "--target", *targets, "--parallel", "4"], output / "compile.log")
    executable = output / "Release/nafx-render.exe" if os.name == "nt" else output / "nafx-render"
    report = {"build": str(output), "executable": str(executable), "executable_sha256": sha256(executable),
              "sdk": str(sdk), "plugin_requested": plugin,
              "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in sorted((REPO / "deploy").rglob("*")) if p.is_file()},
              "environment": environment()}
    write_json(output / "build_manifest.json", report)
    return report


def validator_build(output, sdk=None):
    source = Path(sdk).resolve() if sdk else REPO / ".deps/native/vst3sdk"
    output = Path(output).resolve()
    if not source.exists():
        if not shutil.which("git"):
            raise ValueError("Git is needed to fetch the official validator sources")
        print("Fetching pinned Steinberg SDK...", flush=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/steinbergmedia/vst3sdk.git", source])
        _run(["git", "-C", source, "checkout", VST3_REVISION])
        _run(["git", "-C", source, "submodule", "update", "--init", "--depth", "1", "base", "cmake", "pluginterfaces", "public.sdk"])
    revision = _run(["git", "-C", source, "rev-parse", "HEAD"]).strip()
    if sdk is None and revision != VST3_REVISION:
        raise ValueError("Cached validator SDK differs from pinned revision; provide an explicit --sdk or use a fresh dependency directory")
    command = ["cmake", "-S", source, "-B", output, "-DSMTG_ENABLE_VSTGUI_SUPPORT=OFF", "-DSMTG_ENABLE_VST3_PLUGIN_EXAMPLES=OFF", "-DSMTG_ENABLE_VST3_HOSTING_EXAMPLES=ON"]
    if os.name == "nt":
        command += ["-A", "x64"]
    print("Building official Steinberg validator...", flush=True)
    _run(command, output / "configure.log")
    _run(["cmake", "--build", output, "--config", "Release", "--target", "validator", "--parallel", "4"], output / "compile.log")
    executable = output / "bin/Release/validator.exe" if os.name == "nt" else output / "bin/validator"
    result = {"executable": str(executable), "sha256": sha256(executable), "sdk_revision": revision}
    write_json(output / "build_manifest.json", result)
    return result


def validate_plugin(plugin, validator, output, model=None):
    output, plugin, validator = Path(output), Path(plugin).resolve(), Path(validator).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Validator output already exists: {output}")
    output.mkdir(parents=True)
    cases = []
    for name, model_path in [("default", None)] + ([("loaded_model", Path(model).resolve())] if model else []):
        env = dict(os.environ)
        env.pop("NEURAL_FX_MODEL", None)
        if model_path:
            env["NEURAL_FX_MODEL"] = str(model_path)
        run = subprocess.run([str(validator), str(plugin)], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env, check=False)
        log = run.stdout + run.stderr
        (output / f"{name}.log").write_text(log, encoding="utf-8")
        counts = re.search(r"Result:\s*(\d+)\s+tests?\s+passed,\s*(\d+)\s+tests?\s+failed", log, re.IGNORECASE)
        case = {"case": name, "returncode": run.returncode,
                "tests_passed": int(counts[1]) if counts else None, "tests_failed": int(counts[2]) if counts else None,
                "passed": run.returncode == 0 and counts is not None and int(counts[2]) == 0,
                "model_sha256": sha256(model_path) if model_path else None}
        cases.append(case)
        print(f"Steinberg validator {name}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
    result = {"validator_sha256": sha256(validator), "plugin": str(plugin),
              "plugin_files_sha256": {str(p.relative_to(plugin)): sha256(p) for p in plugin.rglob("*") if p.is_file()},
              "cases": cases, "passed": all(case["passed"] for case in cases), "production_ready": False,
              "scope": "Official VST3 conformity tests; not a DAW host check, fidelity claim or allocation audit"}
    write_json(output / "validator.json", result)
    return result


def native_sustain(model, executable, output, seconds=600):
    output, executable, model = Path(output), Path(executable).resolve(), Path(model).resolve()
    if seconds < 1 or seconds > 3600:
        raise ValueError("Use a sustained duration between 1 and 3600 seconds")
    if output.exists():
        raise FileExistsError(f"Sustained evidence already exists: {output}")
    print(f"Running paced native benchmark for {seconds} seconds...", flush=True)
    _run([executable, "benchmark", "--model", model, "--seconds", seconds, "--paced", "--output", output], output.with_suffix(".log"))
    report = read_json(output)
    report.update({"executable_sha256": sha256(executable), "onnx_sha256": sha256(model), "environment": environment()})
    write_json(output, report)
    return report


def package_preview(validation, output, build="build/native", reference_preset=None, samples="samples/distortion"):
    """Assemble local development artifacts, without installing or publishing them."""
    validation, output, build = Path(validation), Path(output), Path(build)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preview output already exists: {output}")
    evidence = read_json(validation / "validation.json")
    if not evidence["parity_passed"]:
        raise ValueError("Preview requires accepted native parity evidence")
    output.mkdir(parents=True)
    plugin = build / "plugin/NeuralAudioFX_artefacts/Release/VST3/NeuralAudioFX.vst3"
    standalone = build / "plugin/NeuralAudioFX_artefacts/Release/Standalone"
    renderer = build / "Release"
    shutil.copytree(plugin, output / "VST3/NeuralAudioFX.vst3", ignore=shutil.ignore_patterns("onnxruntime.dll"))
    (output / "Standalone").mkdir()
    (output / "Renderer").mkdir()
    for file in ("NeuralAudioFX.exe", "neural_fx_ort_123.dll"):
        shutil.copyfile(standalone / file, output / "Standalone" / file)
    for file in ("nafx-render.exe", "neural_fx_ort_123.dll"):
        shutil.copyfile(renderer / file, output / "Renderer" / file)
    (output / "models").mkdir()
    for row in evidence["blocks"]:
        source = validation / str(row["block_size"]) / "model.onnx"
        if sha256(source) != row["export"]["onnx_sha256"]:
            raise ValueError("Preview model differs from its validation snapshot")
        shutil.copyfile(source, output / "models" / f"conditional_tcn-{row['block_size']}.onnx")
    if not (output / "models/conditional_tcn-128.onnx").exists():
        raise ValueError("Preview launcher requires a validated 128-sample export")
    if reference_preset:
        (output / "presets").mkdir()
        shutil.copyfile(reference_preset, output / "presets/reference-example.json")
    # Original algorithmic notes for an offline renderer demo; no external recordings.
    rate = 48000
    t = np.arange(rate * 9) / rate
    dry = np.zeros(len(t), dtype=np.float32)
    for index, frequency in enumerate((82.41, 110, 146.83, 196, 246.94, 164.81, 130.81, 98, 82.41, 146.83, 196, 110)):
        start, length = int(index * 0.75 * rate), int(0.75 * rate)
        local = np.arange(length) / rate
        envelope = (1 - np.exp(-local * 400)) * np.exp(-local * 4)
        note = sum(np.sin(2 * np.pi * frequency * harmonic * local) / harmonic for harmonic in range(1, 7))
        dry[start:start + length] = (note * envelope * (0.10 + 0.025 * (index % 4))).astype(np.float32)
    wavfile.write(output / "demo-input.wav", rate, dry)
    _run([output / "Renderer/nafx-render.exe", "render", "--model", output / "models/conditional_tcn-128.onnx",
          "--input", output / "demo-input.wav", "--output", output / "demo-rendered.wav", "--drive", "0.5", "--mix", "1"])
    if samples and (Path(samples) / "manifest.json").exists():
        library = read_json(Path(samples) / "manifest.json")
        for name, digest in library["files_sha256"].items():
            file = (Path(samples) / name).resolve()
            if not file.is_relative_to(Path(samples).resolve()) or sha256(file) != digest:
                raise ValueError("Sample pack checksum/path mismatch")
        shutil.copytree(samples, output / "samples")
    pedal_default = bool(samples and (Path(samples) / "manifest.json").exists() and any(c.get("engine") == 1 for c in library["clips"]))
    (output / "Launch-NeuralFX.cmd").write_text('@echo off\nsetlocal\nset "NEURAL_FX_MODEL=%~dp0models\\conditional_tcn-128.onnx"\nset "NEURAL_FX_SAMPLES=%~dp0samples"\nset "NEURAL_FX_PEDAL=' + ('1' if pedal_default else '') + '"\n"%~dp0Standalone\\NeuralAudioFX.exe"\n', encoding="utf-8")
    (output / "README.txt").write_text(
        "NeuralAudioFX local research preview\n\nRun Launch-NeuralFX.cmd to open the standalone app with selectable neural and pedal effects.\n"
        "Set audio-device/session rate to 48000 Hz. Drive and Mix are normalized 0..1.\n"
        "Choose Pedal distortion (4x) for the conventional oversampled guitar effect, or Neural model for learned inference.\n"
        "Crunch/Heavy configure the pedal. Pedal input and tone apply to that mode; Output trims either effect.\n"
        "Select a dataset sample and press Play; Loop repeats it, Dry compare auditions the dry input. Open WAV accepts your own mono/stereo audio (up to 60 seconds).\n"
        "Synthetic packs include DSP targets in samples/*-wet.wav. Guitar DI packs contain dry recordings with effect applied live. Input is replaced only while sample playback is active.\n"
        "Choose model loads another current export outside audio processing. Load reference preset applies inferred controls.\n"
        "The buffered plugin reports one model block of latency (128 samples for the default model).\n"
        "demo-input.wav and demo-rendered.wav are nine-second algorithmic test notes, not real guitar recordings.\n"
        "For a DAW, copy the entire VST3/NeuralAudioFX.vst3 folder into a host-scanned VST3 location and rescan.\n"
        "Keep neural_fx_ort_123.dll inside the bundle. No installation or system settings were changed by packaging.\n\n"
        "Development preview only: native callback scheduling, runtime internals, real captures, listening study and DAW host acceptance remain separate gates.\n"
        "Review the repository THIRD_PARTY_NOTICES.md and SDK licenses before distributing builds.\n", encoding="utf-8")
    notices = output / "licenses"
    notices.mkdir()
    for dependency in ("onnxruntime-win-x64-1.23.2", "JUCE-8.0.12"):
        root = REPO / ".deps/native" / dependency
        for item in root.glob("*LICENSE*"):
            if item.is_file():
                shutil.copyfile(item, notices / f"{dependency}-{item.name}")
        for item in root.glob("*ThirdParty*Notices*"):
            if item.is_file():
                shutil.copyfile(item, notices / f"{dependency}-{item.name}")
    result = {"scope": "local Windows x64 development preview; not installed or published",
              "validation_sha256": sha256(validation / "validation.json"), "production_ready": False,
              "files_sha256": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    write_json(output / "manifest.json", result)
    return {"preview": str(output), "launcher": str(output / "Launch-NeuralFX.cmd"), "production_ready": False}


def _trace_fixture(model, block, destination):
    rng = np.random.default_rng(411)
    cases, expected = [], []
    history, hidden = model.initial_state()
    with torch.inference_mode():
        for index in range(24):
            reset = index in (0, 16)
            # Repeat the first eight blocks after reset to test exact state reset.
            local = index if index < 16 else index - 16
            if index >= 16:
                _, controls, audio = cases[local]
            else:
                audio = rng.uniform(-0.4, 0.4, block).astype(np.float32)
                if index == 0: audio[:] = 0
                if index == 1: audio[:] = 0; audio[0] = 1
                if index == 2: audio[:] = 1
                if index == 3: audio[:] = np.where(np.arange(block) % 2, -1, 1)
                if index == 4: audio[:] = np.sin(np.arange(block) * 0.23)
                controls = np.array([rng.uniform(0, 1), 0 if index in (5, 11) else rng.uniform(0.1, 1)], dtype=np.float32)
            if reset:
                history, hidden = model.initial_state()
            x, c = torch.from_numpy(audio)[None, None], torch.from_numpy(controls)[None]
            y, history, hidden = model.process(x, c, history, hidden)
            cases.append((reset, controls, audio))
            expected.append((y.numpy().ravel().copy(), history.numpy().ravel().copy(), hidden.numpy().ravel().copy() if model.has_gru else np.zeros(0, np.float32)))
    with Path(destination).open("wb") as handle:
        handle.write(struct.pack("<8sII", b"NFXTRACE", len(cases), block))
        for reset, controls, audio in cases:
            handle.write(struct.pack("<Iff", int(reset), *controls))
            handle.write(audio.astype("<f4").tobytes())
    return expected


def _read_trace(path):
    with Path(path).open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise ValueError("Truncated native output header")
        magic, count, block, history, hidden = struct.unpack("<8sIIII", header)
        if magic != b"NFXOUT01" or count < 1 or max(count, block, history, hidden) > 1000000:
            raise ValueError("Invalid native output dimensions")
        data = np.frombuffer(handle.read(), dtype="<f4")
        if data.size != count * (block + history + hidden):
            raise ValueError("Native output length mismatch")
        rows = data.reshape(count, block + history + hidden)
        return [(row[:block], row[block:block + history], row[block + history:]) for row in rows]


def native_validate(checkpoint, executable, output, block_sizes=(64, 128, 256, 512), plugin=None, host_check=None):
    output, executable = Path(output), Path(executable).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Validation output already exists: {output}")
    if not executable.is_file() or not block_sizes or any(size < 1 for size in block_sizes):
        raise ValueError("A built native executable and positive block sizes are required")
    output.mkdir(parents=True)
    seed_everything(1337)
    model, saved = load_checkpoint(checkpoint)
    report = {"checkpoint_sha256": sha256(checkpoint), "executable_sha256": sha256(executable),
              "environment": environment(), "blocks": [], "host_plugin_checks": [], "production_ready": False}
    rate = saved["config"]["data"]["sample_rate"]
    tolerance = saved["config"]["gates"]["export_max_error"]
    write_json(output / "status.json", {"state": "running"})
    try:
        for block in block_sizes:
            folder = output / str(block)
            folder.mkdir()
            exported = export_onnx(checkpoint, folder / "model.onnx", block)
            expected = _trace_fixture(model, block, folder / "input.trace")
            _run([executable, "trace", "--model", folder / "model.onnx", "--input", folder / "input.trace", "--output", folder / "native.trace"], folder / "trace.log")
            actual = _read_trace(folder / "native.trace")
            if len(actual) != len(expected):
                raise ValueError("Trace block count mismatch")
            errors = [max((float(np.max(np.abs(a - b))) for a, b in zip(reference, native) if len(a)), default=0)
                      for reference, native in zip(expected, actual)]
            reset_error = max(float(np.max(np.abs(a - b))) for before, after in zip(actual[:8], actual[16:]) for a, b in zip(before, after) if len(a))
            # Offline and host-buffered WAV paths include a final partial block.
            x = np.random.default_rng(911).uniform(-0.2, 0.2, 1777).astype(np.float32)
            wavfile.write(folder / "input.wav", rate, x)
            controls = torch.tensor([[0.5, 1.0]])
            reference = stream(model, torch.from_numpy(x)[None, None], controls, (block,)).numpy().ravel()
            _run([executable, "render", "--model", folder / "model.onnx", "--input", folder / "input.wav", "--output", folder / "rendered.wav"], folder / "render.log")
            rendered_rate, audio = read_audio(folder / "rendered.wav")
            render_error = float(np.max(np.abs(reference - audio)))
            host_errors = {}
            delayed = np.concatenate((np.zeros(block, np.float32), reference))[:len(x)]
            for host_block in (17, 64, 113, 256, 512):
                path = folder / f"buffered-{host_block}.wav"
                _run([executable, "render", "--buffered", "--host-block", host_block, "--model", folder / "model.onnx", "--input", folder / "input.wav", "--output", path])
                _, buffered = read_audio(path)
                host_errors[str(host_block)] = float(np.max(np.abs(buffered - delayed)))
            _run([executable, "benchmark", "--model", folder / "model.onnx", "--output", folder / "benchmark.json", "--iterations", 2000], folder / "benchmark.log")
            timing = read_json(folder / "benchmark.json")
            row = {"block_size": block, "export": exported, "trace_max_error": max(errors), "reset_max_error": reset_error,
                   "wav_max_error": render_error, "buffered_host_errors": host_errors,
                   "latency_samples": block, "benchmark": timing,
                   "parity_passed": max([*errors, reset_error, render_error, *host_errors.values()]) <= tolerance and rendered_rate == rate}
            report["blocks"].append(row)
            print(f"Native block {block}: parity={'PASS' if row['parity_passed'] else 'FAIL'}, p99={timing['p99_ms']:.4f} ms, application new calls={timing['application_new_calls']}", flush=True)
            write_json(output / "validation.json", report)
        if plugin and host_check:
            # Plugin is explicitly loaded through a real VST3 host wrapper, not linked directly.
            x = np.random.default_rng(911).uniform(-0.2, 0.2, 1777).astype(np.float32)
            x.tofile(output / "host-input.f32")
            env = dict(os.environ)
            # Plugin validation is for a fixed exported block; use the first checked model.
            env["NEURAL_FX_MODEL"] = str((output / str(block_sizes[0]) / "model.onnx").resolve())
            reference = stream(model, torch.from_numpy(x)[None, None], torch.tensor([[0.5, 1.0]]), (block_sizes[0],)).numpy().ravel()
            delayed = np.concatenate((np.zeros(block_sizes[0], np.float32), reference))[:len(x)]
            for host_block in (17, 64, 113, 256, 512):
                path = output / f"plugin-{host_block}.f32"
                log = _run([Path(host_check).resolve(), Path(plugin).resolve(), output / "host-input.f32", path, host_block], output / f"plugin-{host_block}.log", env)
                values = np.fromfile(path, dtype=np.float32)
                if values.shape != delayed.shape:
                    raise ValueError("Plugin host output length mismatch")
                error = float(np.max(np.abs(values - delayed)))
                report["host_plugin_checks"].append({"host_block": host_block, "max_error": error, "passed": error <= tolerance})
        report["parity_passed"] = all(row["parity_passed"] for row in report["blocks"]) and all(row["passed"] for row in report["host_plugin_checks"])
        report["native_screening_passed"] = all(row["benchmark"]["screening_passed"] for row in report["blocks"])
        report["remaining"] = ["sustained paced scheduling check", "ONNX Runtime internal allocation audit", "Ableton/Audacity host checks", "human listening and real captures"]
        write_json(output / "validation.json", report)
        write_json(output / "status.json", {"state": "completed", "parity_passed": report["parity_passed"]})
        if not report["parity_passed"]:
            raise ValueError("Native validation parity failed; inspect validation.json")
        return report
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise
