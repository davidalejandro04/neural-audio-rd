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
from .runtime import benchmark, export_onnx
from .utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Reproducible neural audio-effects experiments")
    commands = parser.add_subparsers(dest="command", required=True)
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
        if args.command == "prepare":
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
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "run" and not result["all_engineering_gates_passed"]:
        parser.exit(1, "One or more engineering gates failed; inspect each models/*/gates.json.\n")
