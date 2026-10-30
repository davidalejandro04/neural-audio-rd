"""Audit the v5 package, check actual trained inference, and preserve evidence.

Run the native UI/effects harness first (see docs/HIGH_GAIN.md). This script
never changes model weights or overwrites historical experiment decisions.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

from neural_fx.data import read_audio
from neural_fx.dsp import target
from neural_fx.native import _run
from neural_fx.utils import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.run / "models/conditional_tcn"
    native = read_json(args.native / "validation.json")
    package = read_json(args.preview / "manifest.json")
    for name, digest in package["files_sha256"].items():
        file = (args.preview / name).resolve()
        if not file.is_relative_to(args.preview.resolve()) or sha256(file) != digest:
            raise ValueError(f"Package checksum/path mismatch: {name}")
    model = args.preview / "models/conditional_tcn-128.onnx"
    if sha256(model) != native["blocks"][0]["export"]["onnx_sha256"]:
        raise ValueError("Packaged model differs from native parity snapshot")
    if sha256(model_dir / "best.pt") != native["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from native parity snapshot")
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata["neural_fx_effect"] != "high_gain" or metadata["neural_fx_mix_contract"] != "output_already_mixed":
        raise ValueError("Unexpected trained model contract")
    history_shape = next(p.shape for p in session.get_inputs() if p.name == "history")
    history = np.zeros(history_shape, np.float32)
    silence_error = 0.
    for drive in (0., .5, 1., .2, .9):
        for _ in range(4):
            y, history = session.run(None, {"audio": np.zeros((1,1,128), np.float32),
                                          "controls": np.array([[drive, 1]], np.float32), "history": history})
            silence_error = max(silence_error, float(np.max(np.abs(y))))
    if silence_error != 0:
        raise ValueError("Trained ONNX model generates output on silence")
    library = read_json(args.preview / "samples/manifest.json")
    listening = args.checks / "listening"; listening.mkdir(exist_ok=True)
    guitar = []
    for clip in library["clips"]:
        path = args.preview / "samples" / clip["dry"]
        rate, dry = read_audio(path)
        stem = path.stem.removesuffix("-dry")
        row = {"source": stem, "rack_renders": {}, "synthetic_target_transfer": []}
        for preset in ("british-lead", "neural-lead"):
            wet = np.fromfile(args.checks / f"{stem}-{preset}.f32", np.float32)[128:]
            if len(wet) != len(dry) + 96000 or not np.isfinite(wet).all():
                raise ValueError("Missing or invalid full-rig guitar render")
            peak = float(np.max(np.abs(wet)))
            if peak >= 1:
                raise ValueError("Guitar preset clips full scale")
            wavfile.write(listening / f"{stem}-{preset}.wav", rate, wet)
            row["rack_renders"][preset] = {"peak": peak, "rms": float(np.sqrt(np.mean(wet**2))),
                                             "wav_sha256": sha256(listening / f"{stem}-{preset}.wav")}
        # First second only: unseen real DI input against our synthetic target,
        # not an evaluation against real amplifier audio.
        x = dry[:48000]
        for drive in (.3, .85):
            history = np.zeros(history_shape, np.float32); blocks = []
            for start in range(0, len(x), 128):
                block = np.pad(x[start:start+128], (0, max(0, start+128-len(x))))
                y, history = session.run(None, {"audio": block[None,None], "controls": np.array([[drive,1]], np.float32), "history": history})
                blocks.append(y.ravel())
            prediction = np.concatenate(blocks)[:len(x)]
            reference = target(x, [drive,1], "high_gain", rate)
            esr = float(np.sum((prediction-reference)**2) / np.sum(reference**2))
            row["synthetic_target_transfer"].append({"drive": drive, "samples": len(x), "esr": esr})
        guitar.append(row)
    # Check the copied VST3, not only the build-tree plugin.
    import os
    env = dict(os.environ, NEURAL_FX_MODEL=str(model.resolve()), NEURAL_FX_PEDAL="", NEURAL_FX_RIG="")
    host = Path("build/native/plugin/nafx-plugin-check_artefacts/Release/nafx-plugin-check.exe").resolve()
    packaged_host = []
    for drive, mix in ((.5,1.), (.85,.35), (.85,0.)):
        destination = args.checks / f"packaged-{drive}-{mix}.f32"
        _run([host, (args.preview / "VST3/NeuralAudioFX.vst3").resolve(), (args.native / "host-input.f32").resolve(), destination.resolve(), 113, drive, mix], env=env)
        expected = np.fromfile(args.native / f"plugin-113-{drive}-{mix}.f32", np.float32)
        actual = np.fromfile(destination, np.float32)
        error = float(np.max(np.abs(actual-expected)))
        if error > 1e-6:
            raise ValueError("Packaged VST3 differs from validated build")
        packaged_host.append({"drive":drive, "mix":mix, "max_error":error})
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Synthetic high-gain model plus independent native guitar rack; no hardware/artist fidelity claim",
        "checkpoint_sha256": sha256(model_dir / "best.pt"), "onnx_sha256": sha256(model),
        "configuration": read_json(args.run / "preregistered_config.json"),
        "evaluation": read_json(model_dir / "evaluation/metrics.json"),
        "engineering_gates": read_json(model_dir / "gates.json"),
        "native": {key:native[key] for key in ("parity_passed", "native_screening_passed", "blocks", "host_plugin_checks")},
        "pedal_and_guitar_callback": read_json(args.checks / "processor-checks.json"),
        "rack": read_json(args.checks / "rack-checks.json"), "trained_onnx_silence_max": silence_error,
        "guitar": guitar, "packaged_host": packaged_host, "package_manifest_sha256": sha256(args.preview / "manifest.json"),
        "official_validator": read_json(args.validator / "validator.json"),
        "screenshots": {str(path):sha256(path) for path in (Path("docs/assets/hard-rock-rig.png"), Path("docs/assets/high-gain-neural.png"))},
        "production_ready": False,
        "limitations": ["Python runtime gate failed and is retained", "Native timing is an unpaced screen, not sustained device acceptance",
                        "Guitar transfer compares with synthetic DSP, not real amp captures", "No human listening or target DAW/device acceptance",
                        "Cabinets are algorithmic filters; no measured IRs", "No complete aliasing or external-runtime allocation audit"]}
    write_json(args.output, result)
    print({"model_silence_max":silence_error, "guitar_clips":len(guitar), "packaged_host_cases":len(packaged_host), "evidence":str(args.output)})


if __name__ == "__main__":
    main()
