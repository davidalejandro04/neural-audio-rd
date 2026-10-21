"""Train on train, select on validation, evaluate once on held-out sources."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader

from .data import PairedDataset, verify
from .dsp import metrics, mrstft, target
from .models import build_model, stream
from .utils import canonical_hash, environment, read_json, seed_everything, sha256, write_json


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def train(config, root, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run already exists: {output}; use a new run directory")
    audit = verify(root)
    if audit["sample_rate"] != config["data"]["sample_rate"] or audit["effect"] != config["data"]["effect"]:
        raise ValueError("Dataset does not match experiment sample rate/effect")
    seed_everything(config["seed"])
    model = build_model(config["model"])
    settings = config["training"]
    if settings["epochs"] < 1:
        raise ValueError("epochs must be positive")
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    write_json(output / "dataset_audit.json", audit)
    dataset = PairedDataset(root, "train", settings["segment_samples"], model.history_samples)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True, num_workers=0)
    validation = PairedDataset(root, "valid")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    best, history = float("inf"), []
    for epoch in range(settings["epochs"]):
        model.train()
        total = 0.0
        for dry, wet, controls in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(dry, controls)[..., -wet.shape[-1]:]
            loss = (prediction - wet).square().mean() + settings["spectral_weight"] * mrstft(prediction, wet)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(dry)
        model.eval()
        with torch.inference_mode():
            # Include active settings only so exact bypass cannot inflate selection.
            scores = [metrics(model(dry[None], controls[None]), wet[None])["esr"]
                      for dry, wet, controls in validation if controls[1] > 0]
        if not scores or any(score is None for score in scores):
            raise ValueError("Validation needs non-silent active-effect examples")
        score = float(np.mean(scores))
        history.append({"epoch": epoch + 1, "train_loss": total / len(dataset), "valid_esr": score})
        if score < best:
            best = score
            torch.save({"state_dict": model.state_dict(), "model_spec": config["model"],
                        "config": config, "epoch": epoch + 1,
                        "config_sha256": canonical_hash(config),
                        "manifest_sha256": audit["manifest_sha256"]}, output / "best.pt")
    write_json(output / "training.json", history)
    return {"best_valid_esr": best, "epochs": len(history), "checkpoint_sha256": sha256(output / "best.pt")}


def aggregate(rows):
    names = ("mse", "l1", "esr", "mrstft", "lsd_db")
    return {name: float(np.median([row[name] for row in rows if row[name] is not None]))
            if any(row[name] is not None for row in rows) else None for name in names}


@torch.inference_mode()
def evaluate(checkpoint_path, root, output):
    seed_everything(0)
    model, checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    audit = verify(root)
    if audit["manifest_sha256"] != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from the training snapshot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = PairedDataset(root, "test")
    rows, max_stream_error = [], 0.0
    synthetic = (Path(root) / "dataset.json").exists()
    for index, (dry, wet, controls) in enumerate(dataset):
        entry = dataset.entries[index]
        x, y, c = dry[None], wet[None], controls[None]
        prediction = model(x, c)
        for block in config["evaluation"]["block_sizes"]:
            error = (stream(model, x, c, (block,)) - prediction).abs().max().item()
            max_stream_error = max(max_stream_error, error)
        max_stream_error = max(max_stream_error, (stream(model, x, c, (17, 64, 113, 256)) - prediction).abs().max().item())
        candidates = {"model": prediction, "identity": x}
        if synthetic:
            oracle = target(dry.numpy().ravel(), [entry["parameters"]["drive"], entry["parameters"]["mix"]], entry["effect_type"], entry["sample_rate"])
            candidates["analytic_oracle"] = torch.from_numpy(oracle)[None, None]
        for name, audio in candidates.items():
            rows.append({"pair_id": entry["pair_id"], "source_id": entry["source_id"],
                         "regime": entry.get("regime", "unspecified"), "system": name,
                         "bypass": controls[1].item() == 0, **metrics(audio, y)})
        if index < config["evaluation"]["audio_examples"]:
            folder = output / "audio" / entry["pair_id"]
            folder.mkdir(parents=True, exist_ok=True)
            for name, audio in {"dry": x, "target": y, "model": prediction}.items():
                wavfile.write(folder / f"{name}.wav", entry["sample_rate"], audio.numpy().ravel())
    summary = {}
    for system in sorted({row["system"] for row in rows}):
        active = [row for row in rows if row["system"] == system and not row["bypass"]]
        # Average presets within a source, then median over independent sources.
        source_scores = []
        for source in sorted({row["source_id"] for row in active}):
            selected = [row for row in active if row["source_id"] == source]
            source_scores.append({key: float(np.mean([row[key] for row in selected if row[key] is not None]))
                                  if any(row[key] is not None for row in selected) else None
                                  for key in ("mse", "l1", "esr", "mrstft", "lsd_db")})
        summary[system] = {"active_source_median": aggregate(source_scores),
                           "by_regime_pair_median": {regime: aggregate([row for row in active if row["regime"] == regime])
                                                     for regime in sorted({row["regime"] for row in active})}}
    report = {"checkpoint_sha256": sha256(checkpoint_path), "config_sha256": checkpoint["config_sha256"],
              "manifest_sha256": audit["manifest_sha256"], "independent_test_sources": len({r["source_id"] for r in rows}),
              "max_stream_error": max_stream_error, "parameters": sum(p.numel() for p in model.parameters()),
              "history_samples": model.history_samples, "summary": summary,
              "scope": "synthetic engineering check" if synthetic else "paired capture evaluation"}
    write_json(output / "metrics.json", report)
    write_json(output / "per_pair.json", rows)
    return report


def compare(run_root, output):
    reports = []
    for path in sorted(Path(run_root).glob("*/evaluation/metrics.json")):
        report = read_json(path)
        reports.append({"model": path.parents[1].name, "parameters": report["parameters"],
                        "max_stream_error": report["max_stream_error"],
                        **report["summary"]["model"]["active_source_median"]})
    if not reports:
        raise ValueError("No evaluation reports found")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", reports)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    return reports
