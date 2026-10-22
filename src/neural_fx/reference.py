"""Paired-reference control inference baseline with a source-consistency latent.

The decoder is explicit DSP during baseline evaluation. A frozen conditional
audio checkpoint can also consume inferred controls at render time. This is not
a jointly trained reference-conditioned hybrid or evidence of perceptual parity.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.io import wavfile

from .data import aligned_pair, generate, read_audio, records, verify
from .dsp import metrics, target
from .experiment import load_checkpoint
from .models import stream
from .utils import environment, read_json, seed_everything, sha256, write_json


def pair_features(dry, wet):
    dry, wet = np.asarray(dry, np.float64), np.asarray(wet, np.float64)
    if dry.ndim != 1 or wet.shape != dry.shape or len(dry) < 256 or not np.isfinite(dry).all() or not np.isfinite(wet).all():
        raise ValueError("References must be aligned finite mono arrays, at least 256 samples")
    if np.mean(dry ** 2) < 1e-10:
        raise ValueError("Reference dry signal is silent; effect controls cannot be inferred")
    eps = 1e-8
    rms_x, rms_y = np.sqrt(np.mean(dry ** 2)), np.sqrt(np.mean(wet ** 2))
    norm = rms_x * rms_y + eps
    values = [np.log(rms_x + eps), np.log(rms_y + eps), np.log((rms_y + eps) / (rms_x + eps)),
              np.mean(dry * wet) / norm, np.mean(np.abs(wet - dry)) / (rms_x + eps),
              np.sqrt(np.mean((wet - dry) ** 2)) / (rms_x + eps),
              np.mean(dry), np.mean(wet), np.max(np.abs(dry)), np.max(np.abs(wet)),
              np.mean(np.abs(dry)) / (rms_x + eps), np.mean(np.abs(wet)) / (rms_y + eps)]
    x = np.abs(np.fft.rfft(dry)) ** 2
    y = np.abs(np.fft.rfft(wet)) ** 2
    for a, b in zip(np.array_split(x, 8), np.array_split(y, 8)):
        values.append(np.log((b.mean() + eps) / (a.mean() + eps)))
    values.extend([np.mean(dry ** 4) / (rms_x ** 4 + eps), np.mean(wet ** 4) / (rms_y ** 4 + eps)])
    return np.asarray(values, dtype=np.float32)


class ReferenceEncoder(nn.Module):
    def __init__(self, features=22, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, 64), nn.Tanh(), nn.Linear(64, latent_dim))
        self.controls = nn.Sequential(nn.Tanh(), nn.Linear(latent_dim, 2), nn.Sigmoid())

    def forward(self, features):
        latent = self.encoder(features)
        return self.controls(latent), latent


def _split_features(root, split):
    entries = [entry for entry in records(root, split) if entry["parameters"]["mix"] > 0]
    if not entries:
        raise ValueError("Reference inference requires active effect examples")
    features, labels = [], []
    for entry in entries:
        dry, wet = aligned_pair(root, entry)
        features.append(pair_features(dry, wet))
        labels.append([entry["parameters"]["drive"], entry["parameters"]["mix"]])
    return entries, torch.from_numpy(np.stack(features)), torch.tensor(labels)


def load_reference(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    encoder = ReferenceEncoder(latent_dim=saved["latent_dim"])
    encoder.load_state_dict(saved["state_dict"])
    return encoder.eval(), saved


def train_reference(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Reference run already exists: {output}")
    seed_everything(config["seed"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    root = output / "dataset"
    audit = generate(config, root)
    train_entries, x, y = _split_features(root, "train")
    _, vx, vy = _split_features(root, "valid")
    mean, scale = x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x, vx = (x - mean) / scale, (vx - mean) / scale
    settings = config["reference"]
    if settings["epochs"] < 1:
        raise ValueError("Reference epochs must be positive")
    encoder = ReferenceEncoder(latent_dim=settings["latent_dim"])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=settings["learning_rate"])
    groups = {}
    for index, entry in enumerate(train_entries):
        key = (entry["device_id"], entry["parameters"]["drive"], entry["parameters"]["mix"])
        groups.setdefault(key, []).append(index)
    history, best = [], float("inf")
    for epoch in range(settings["epochs"]):
        encoder.train()
        predicted, latent = encoder(x)
        reconstruction = (predicted - y).square().mean()
        consistency = torch.stack([latent[indices].var(dim=0, unbiased=False).mean() for indices in groups.values()]).mean()
        loss = reconstruction + settings["consistency_weight"] * consistency
        if not torch.isfinite(loss):
            raise ValueError("Reference training diverged")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        encoder.eval()
        with torch.inference_mode():
            score = (encoder(vx)[0] - vy).square().mean().item()
        history.append({"epoch": epoch + 1, "train_control_mse": reconstruction.item(), "consistency": consistency.item(), "valid_control_mse": score})
        if score < best:
            best = score
            torch.save({"state_dict": encoder.state_dict(), "mean": mean, "scale": scale,
                        "latent_dim": settings["latent_dim"], "config": config,
                        "manifest_sha256": audit["manifest_sha256"], "epoch": epoch + 1}, output / "reference.pt")
    write_json(output / "training.json", history)
    result = evaluate_reference(output / "reference.pt", root, output / "evaluation")
    return {"checkpoint": str(output / "reference.pt"), "best_validation_control_mse": best, "evaluation": result}


@torch.inference_mode()
def evaluate_reference(checkpoint, root, output):
    encoder, saved = load_reference(checkpoint)
    if verify(root)["manifest_sha256"] != saved["manifest_sha256"]:
        raise ValueError("Reference dataset differs from training snapshot")
    entries, features, labels = _split_features(root, "test")
    controls, latent = encoder((features - saved["mean"]) / saved["scale"])
    training_entries = [entry for entry in records(root, "train") if entry["parameters"]["mix"] > 0]
    mean_controls = np.mean([[entry["parameters"]["drive"], entry["parameters"]["mix"]] for entry in training_entries], axis=0).tolist()
    trials = []
    examples = {}
    for index, entry in enumerate(entries):
        examples.setdefault(entry["source_id"], index)
    if len(examples) < 2:
        raise ValueError("Cross-content testing needs at least two held-out sources")
    source_ids = sorted(examples)
    for index, entry in enumerate(entries):
        other = source_ids[(source_ids.index(entry["source_id"]) + 1) % len(source_ids)]
        dry, _ = aligned_pair(root, entries[examples[other]])
        expected_controls = labels[index].tolist()
        expected = target(dry, expected_controls, entry["effect_type"], entry["sample_rate"])
        predicted = target(dry, controls[index].tolist(), entry["effect_type"], entry["sample_rate"])
        baseline = target(dry, [0.5, 1.0], entry["effect_type"], entry["sample_rate"])
        mean_baseline = target(dry, mean_controls, entry["effect_type"], entry["sample_rate"])
        y = torch.from_numpy(expected)[None, None]
        trials.append({"reference_pair": entry["pair_id"], "reference_source": entry["source_id"], "target_source": other,
                       "target_controls": expected_controls, "inferred_controls": controls[index].tolist(),
                       "reference_system": metrics(torch.from_numpy(predicted)[None, None], y),
                       "no_reference": metrics(torch.from_numpy(baseline)[None, None], y),
                       "training_mean_no_reference": metrics(torch.from_numpy(mean_baseline)[None, None], y)})
    # Independent-unit summaries first aggregate all reference/settings trials for each target source.
    summaries = {}
    for system in ("reference_system", "no_reference", "training_mean_no_reference"):
        summaries[system] = {metric: float(np.median([np.mean([trial[system][metric] for trial in trials if trial["target_source"] == source]) for source in source_ids])) for metric in ("esr", "mrstft")}
    rng = np.random.default_rng(997)
    successes = 0
    for index, entry in enumerate(entries):
        same = [i for i, candidate in enumerate(entries) if candidate["source_id"] != entry["source_id"] and candidate["parameters"] == entry["parameters"]]
        different = [i for i, candidate in enumerate(entries) if candidate["source_id"] != entry["source_id"] and candidate["parameters"] != entry["parameters"]]
        if not same or not different:
            raise ValueError("Latent consistency testing needs repeated and distinct settings on multiple sources")
        successes += torch.linalg.vector_norm(latent[index] - latent[rng.choice(same)]).item() < torch.linalg.vector_norm(latent[index] - latent[rng.choice(different)]).item()
    result = {"scope": "Synthetic paired-reference control inference with known DSP decoder; exploratory P4 baseline",
              "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": saved["manifest_sha256"],
              "test_control_mse": (controls - labels).square().mean().item(), "cross_content_trials": len(trials),
              "independent_test_sources": len(source_ids), "summary": summaries,
              "training_mean_controls": mean_controls,
              "same_effect_latent_closer_fraction": successes / len(entries),
              "production_ready": False, "listening_results": None,
              "limitations": "No joint audio-backbone training, no real captures, no subjective study; estimates require paired and already aligned reference dry/wet audio."}
    write_json(Path(output) / "metrics.json", result)
    write_json(Path(output) / "trials.json", trials)
    return result


@torch.inference_mode()
def infer_reference(checkpoint, dry_path, wet_path, output):
    encoder, saved = load_reference(checkpoint)
    rate, dry = read_audio(dry_path)
    wet_rate, wet = read_audio(wet_path)
    if rate != wet_rate or rate != saved["config"]["data"]["sample_rate"]:
        raise ValueError("Reference and encoder sample rates must match")
    features = torch.from_numpy(pair_features(dry, wet))[None]
    controls, latent = encoder((features - saved["mean"]) / saved["scale"])
    result = {"drive": controls[0, 0].item(), "mix": controls[0, 1].item(), "latent": latent[0].tolist(),
              "sample_rate": rate, "effect": saved["config"]["data"]["effect"],
              "encoder_sha256": sha256(checkpoint), "reference_dry_sha256": sha256(dry_path), "reference_wet_sha256": sha256(wet_path)}
    write_json(output, result)
    return result


def render_reference(encoder_path, audio_checkpoint, reference_dry, reference_wet, input_path, output):
    torch.set_num_threads(1)
    output = Path(output)
    preset = infer_reference(encoder_path, reference_dry, reference_wet, output.with_suffix(".preset.json"))
    model, saved = load_checkpoint(audio_checkpoint)
    rate, dry = read_audio(input_path)
    if rate != preset["sample_rate"] or rate != saved["config"]["data"]["sample_rate"] or preset["effect"] != saved["config"]["data"]["effect"]:
        raise ValueError("Reference encoder, audio backbone and input must share effect family and sample rate")
    controls = torch.tensor([[preset["drive"], preset["mix"]]])
    rendered = stream(model, torch.from_numpy(dry)[None, None], controls).numpy().ravel()
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, rate, rendered)
    return {"output": str(output), "preset": preset, "audio_checkpoint_sha256": sha256(audio_checkpoint), "samples": len(rendered)}
