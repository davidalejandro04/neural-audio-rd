"""Verify real-guitar audition processing and write listenable dry/processed examples.

Run from the repository root after native-build --plugin and package-preview.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from neural_fx.data import read_audio
from neural_fx.native import _run
from neural_fx.utils import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, default=Path("exports/preview-v4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("build/native"))
    args = parser.parse_args()
    pack, output, build = args.preview.resolve(), args.output.resolve(), args.build.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new validation output directory")
    output.mkdir(parents=True, exist_ok=True)
    samples = read_json(pack / "samples/manifest.json")
    if samples.get("license") != "CC0-1.0":
        raise ValueError("Expected the pinned real-guitar pack")
    for name, checksum in samples["files_sha256"].items():
        source = (pack / "samples" / name).resolve()
        if not source.is_relative_to(pack / "samples") or sha256(source) != checksum:
            raise ValueError("Guitar sample checksum/path mismatch")
    model = pack / "models/conditional_tcn-128.onnx"
    plugin = pack / "VST3/NeuralAudioFX.vst3"
    env = dict(os.environ, NEURAL_FX_MODEL=str(model), NEURAL_FX_SAMPLES=str(pack / "samples"),
               NEURAL_FX_PEDAL="1", NEURAL_FX_SCREENSHOT=str(output / "editor.png"),
               NEURAL_FX_EFFECT_CHECK=str(output / "processor"))
    ui = json.loads(_run([build / "plugin/nafx-ui-check_artefacts/Release/nafx-ui-check.exe"], output / "ui.log", env))
    processor = read_json(output / "processor/processor-checks.json")
    latency = processor["latency_samples"]
    audition = output / "audition"
    audition.mkdir()
    metrics, host_checks = [], []
    host = build / "plugin/nafx-plugin-check_artefacts/Release/nafx-plugin-check.exe"
    for index, clip in enumerate(samples["clips"]):
        stem = Path(clip["dry"]).stem.removesuffix("-dry")
        _, dry = read_audio(pack / "samples" / clip["dry"])
        wavfile.write(audition / f"{stem}-dry.wav", 48000, dry)
        row = {"source": stem, "presets": {}}
        for preset in ("light", "crunch", "heavy"):
            wet = np.fromfile(output / f"processor/{stem}-{preset}.f32", np.float32)[latency:]
            if wet.shape != dry.shape or not np.isfinite(wet).all():
                raise ValueError("Invalid guitar render")
            x, y = dry[4096:].astype(np.float64), wet[4096:].astype(np.float64)
            fitted_gain = np.dot(x, y) / np.dot(x, x)
            residual = float(np.linalg.norm(y - fitted_gain * x) / np.linalg.norm(y))
            peak, rms = float(np.max(np.abs(wet))), float(np.sqrt(np.mean(wet.astype(np.float64) ** 2)))
            if not peak < .95 or not .005 < rms < .5 or abs(float(wet.mean())) > .001:
                raise ValueError("Preset clipping, level or DC check failed")
            row["presets"][preset] = {"peak": peak, "rms": rms, "gain_fitted_residual_ratio": residual}
            wavfile.write(audition / f"{stem}-{preset}.wav", 48000, wet)
        if row["presets"]["heavy"]["gain_fitted_residual_ratio"] < row["presets"]["light"]["gain_fitted_residual_ratio"] + .15:
            raise ValueError("Heavy does not add sufficient waveform change beyond gain")
        metrics.append(row)
        for block in ((17, 64, 113, 256, 512) if index == 0 else (113,)):
            dest = output / f"host-{index}-{block}.f32"
            check = json.loads(_run([host, plugin, output / f"processor/{stem}-dry.f32", dest, block], env=env))
            expected = np.fromfile(output / f"processor/{stem}-crunch.f32", np.float32)
            actual = np.fromfile(dest, np.float32)
            if actual.shape != expected.shape:
                raise ValueError("VST3 guitar output length mismatch")
            error = float(np.max(np.abs(actual - expected)))
            if error > 2e-6:
                raise ValueError("VST3 differs from the checked pedal path")
            host_checks.append({**check, "source": stem, "host_block": block, "max_error": error})
    harmonic = {}
    for preset in ("light", "heavy"):
        signal = np.fromfile(output / f"processor/sine-{preset}.f32", np.float32)[latency:][12000:36000]
        spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
        harmonic[preset] = float(np.sqrt(sum(spectrum[k * 500] ** 2 for k in range(2, 20))) / spectrum[500])
    if harmonic["heavy"] < .15 or harmonic["heavy"] < 10 * harmonic["light"]:
        raise ValueError("Harmonic-generation check failed")
    validators = {}
    for mode in ("neural", "pedal"):
        run_env = dict(env, NEURAL_FX_PEDAL="1" if mode == "pedal" else "")
        _run([sys.executable, "-m", "neural_fx", "validate-plugin", "--plugin", plugin,
              "--model", model, "--output", output / f"validator-{mode}"], output / f"validator-{mode}.log", run_env)
        validators[mode] = read_json(output / f"validator-{mode}/validator.json")
    result = {"scope": "Real CC0 DI inputs through a conventional DSP pedal; separate from neural-model research acceptance",
              "passed": True, "production_ready": False, "source_page": samples["source_page"],
              "sample_manifest_sha256": sha256(pack / "samples/manifest.json"), "ui": ui, "processor": processor,
              "guitar_metrics": metrics, "sine_1khz_thd_ratio": harmonic, "vst3_host_checks": host_checks,
              "official_validator": validators,
              "limits": "Not a listening study, hardware-pedal match, target-DAW/device check, complete allocation audit or sustained scheduling acceptance."}
    write_json(output / "validation.json", result)
    print(json.dumps({"passed": True, "guitar_clips": len(metrics), "vst3_host_cases": len(host_checks),
                      "sine_thd": harmonic, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
