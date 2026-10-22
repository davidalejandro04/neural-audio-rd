"""Run with `python -m neural_fx` or the installed `neural-fx` command."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from .data import generate, read_audio, verify
from .experiment import compare, evaluate, load_checkpoint, train
from .models import stream
from .prepare import prepare_pairs
from .listening import package_abx, score_abx
from .pipeline import run
from .runtime import benchmark, benchmark_onnx, benchmark_study, export_onnx
from .study import analyze_study, plan_study, review_run
from .utils import read_json, write_json
from .native import native_build, native_validate, validator_build, validate_plugin, native_sustain, package_preview
from .reference import train_reference, infer_reference, render_reference


def main():
    parser = argparse.ArgumentParser(description="Reproducible neural audio-effects experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("package-preview")
    command.add_argument("--validation", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--build", type=Path, default=Path("build/native"))
    command.add_argument("--reference-preset", type=Path)
    command = commands.add_parser("validator-build")
    command.add_argument("--output", type=Path, default=Path("build/vst3-validator"))
    command.add_argument("--sdk", type=Path)
    command = commands.add_parser("validate-plugin")
    command.add_argument("--plugin", required=True, type=Path)
    command.add_argument("--validator", type=Path, default=Path("build/vst3-validator/bin/Release/validator.exe"))
    command.add_argument("--model", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("native-sustain")
    command.add_argument("--model", required=True, type=Path)
    command.add_argument("--executable", type=Path, default=Path("build/native/Release/nafx-render.exe"))
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--seconds", type=int, default=600)
    command = commands.add_parser("train-reference")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    for name in ("infer-reference", "render-reference"):
        command = commands.add_parser(name)
        command.add_argument("--encoder", required=True, type=Path)
        command.add_argument("--reference-dry", required=True, type=Path)
        command.add_argument("--reference-wet", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "render-reference":
            command.add_argument("--checkpoint", required=True, type=Path)
            command.add_argument("--input", required=True, type=Path)
    command = commands.add_parser("native-build")
    command.add_argument("--output", type=Path, default=Path("build/native"))
    command.add_argument("--sdk", type=Path)
    command.add_argument("--plugin", action="store_true")
    command.add_argument("--juce", type=Path)
    command = commands.add_parser("native-validate")
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--executable", type=Path, default=Path("build/native/Release/nafx-render.exe"))
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--block-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    command.add_argument("--plugin", type=Path)
    command.add_argument("--host-check", type=Path)
    command = commands.add_parser("benchmark-onnx")
    command.add_argument("--model", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--checkpoint", type=Path)
    command.add_argument("--iterations", type=int, default=2000)
    command = commands.add_parser("benchmark-study")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--iterations", type=int, default=2000)
    command = commands.add_parser("review-run")
    command.add_argument("--run", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("plan-study")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("analyze-study")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("prepare")
    command.add_argument("--index", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("abx-package")
    command.add_argument("--evaluation", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--seed", type=int, default=1337)
    command = commands.add_parser("abx-score")
    command.add_argument("--answers", required=True, type=Path)
    command.add_argument("--responses", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    for name in ("generate", "train", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "train":
            command.add_argument("--data", required=True, type=Path)
        if name == "run":
            command.add_argument("--suite", action="store_true")
            command.add_argument("--export", action="store_true")
    command = commands.add_parser("verify")
    command.add_argument("--data", required=True, type=Path)
    for name in ("evaluate", "benchmark", "export", "render"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--data", required=True, type=Path)
        if name in ("export", "render"):
            command.add_argument("--block-size", type=int, default=128)
        if name == "benchmark":
            command.add_argument("--iterations", type=int, default=200)
        if name == "render":
            command.add_argument("--input", required=True, type=Path)
            command.add_argument("--drive", type=float, default=0.5)
            command.add_argument("--mix", type=float, default=1.0)
    command = commands.add_parser("compare")
    command.add_argument("--runs", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "package-preview":
            result = package_preview(args.validation, args.output, args.build, args.reference_preset)
        elif args.command == "validator-build":
            result = validator_build(args.output, args.sdk)
        elif args.command == "validate-plugin":
            result = validate_plugin(args.plugin, args.validator, args.output, args.model)
        elif args.command == "native-sustain":
            result = native_sustain(args.model, args.executable, args.output, args.seconds)
        elif args.command == "train-reference":
            result = train_reference(read_json(args.config), args.output)
        elif args.command == "infer-reference":
            result = infer_reference(args.encoder, args.reference_dry, args.reference_wet, args.output)
        elif args.command == "render-reference":
            result = render_reference(args.encoder, args.checkpoint, args.reference_dry, args.reference_wet, args.input, args.output)
        elif args.command == "native-build":
            result = native_build(args.output, args.sdk, args.plugin, args.juce)
        elif args.command == "native-validate":
            result = native_validate(args.checkpoint, args.executable, args.output, args.block_sizes, args.plugin, args.host_check)
        elif args.command == "benchmark-onnx":
            result = benchmark_onnx(args.model, args.output, args.iterations, args.checkpoint)
        elif args.command == "benchmark-study":
            result = benchmark_study(args.plan, args.output, args.iterations)
        elif args.command == "review-run":
            result = review_run(args.run, args.output)
        elif args.command == "plan-study":
            result = plan_study(read_json(args.config), args.output)
        elif args.command == "analyze-study":
            result = analyze_study(args.plan, args.output)
        elif args.command == "prepare":
            result = prepare_pairs(args.index, args.output)
        elif args.command == "abx-package":
            result = package_abx(args.evaluation, args.output, args.seed)
        elif args.command == "abx-score":
            result = score_abx(args.answers, args.responses, args.output)
        elif args.command == "generate":
            result = generate(read_json(args.config), args.output)
        elif args.command == "verify":
            result = verify(args.data)
        elif args.command == "train":
            result = train(read_json(args.config), args.data, args.output)
        elif args.command == "evaluate":
            result = evaluate(args.checkpoint, args.data, args.output)
        elif args.command == "benchmark":
            result = benchmark(args.checkpoint, args.output, iterations=args.iterations)
        elif args.command == "export":
            result = export_onnx(args.checkpoint, args.output, args.block_size)
        elif args.command == "compare":
            result = compare(args.runs, args.output)
        elif args.command == "run":
            result = run(read_json(args.config), args.output, args.suite, args.export)
        elif args.command == "render":
            if not 0 <= args.drive <= 1 or not 0 <= args.mix <= 1:
                raise ValueError("drive and mix must be in [0, 1]")
            model, checkpoint = load_checkpoint(args.checkpoint)
            rate, audio = read_audio(args.input)
            if rate != checkpoint["config"]["data"]["sample_rate"]:
                raise ValueError("Input sample rate must match model; resample explicitly first")
            torch.set_num_threads(1)
            rendered = stream(model, torch.from_numpy(audio)[None, None],
                              torch.tensor([[args.drive, args.mix]]), (args.block_size,)).numpy().ravel()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(args.output, rate, rendered.astype(np.float32))
            result = {"output": str(args.output), "samples": len(rendered), "peak": float(np.abs(rendered).max())}
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "run" and not result["all_engineering_gates_passed"]:
        parser.exit(1, "One or more engineering gates failed; inspect each models/*/gates.json.\n")
    if args.command == "validate-plugin" and not result["passed"]:
        parser.exit(1, "VST3 conformity tests failed; inspect validator.json and logs.\n")
