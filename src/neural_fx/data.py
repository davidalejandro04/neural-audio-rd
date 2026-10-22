"""Float-WAV paired data, source-level splits, and integrity validation."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from .dsp import target
from .utils import canonical_hash, read_json, sha256, write_json

SPLITS = ("train", "valid", "test")


def read_audio(path):
    rate, samples = wavfile.read(path)
    if samples.dtype.kind in "iu":
        info = np.iinfo(samples.dtype)
        scale = (info.max + 1) / 2 if samples.dtype == np.uint8 else max(abs(info.min), info.max + 1)
        samples = (samples.astype(np.float64) - (128 if samples.dtype == np.uint8 else 0)) / scale
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Expected finite, nonempty mono audio: {path}")
    return rate, samples


def aligned_pair(root, record):
    rate, dry = read_audio(Path(root) / record["dry"])
    wet_rate, wet = read_audio(Path(root) / record["wet"])
    if rate != wet_rate or rate != record["sample_rate"]:
        raise ValueError(f"Sample-rate mismatch: {record['pair_id']}")
    lag = record["latency_samples"]
    if not isinstance(lag, int):
        raise ValueError("latency_samples must be an integer")
    if lag >= 0:
        wet = wet[lag:]
    else:
        dry = dry[-lag:]
    length = min(len(dry), len(wet))
    if length == 0:
        raise ValueError("Latency removes all audio")
    return dry[:length], wet[:length]


def records(root, split):
    return [json.loads(line) for line in (Path(root) / "manifests" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(config, root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Dataset already exists: {root}; use a new output directory")
    settings = config["data"]
    sample_rate, samples = settings["sample_rate"], settings["samples"]
    if sample_rate < 8000 or samples < 256:
        raise ValueError("Use sample_rate >= 8000 and samples >= 256")
    data_seed = config.get("data_seed", config["seed"])
    rng = np.random.default_rng(data_seed)
    default_controls = [(0.1, 1.0), (0.5, 1.0), (0.9, 1.0), (0.5, 0.0)]
    specified = settings.get("controls", {})
    train_controls = specified.get("train", default_controls)
    control_sets = {split: specified.get(split, train_controls if split != "test" else list(train_controls) + [(0.3, 0.65), (0.7, 0.4)]) for split in SPLITS}
    for controls in control_sets.values():
        if not controls or any(len(pair) != 2 or any(not np.isfinite(value) or not 0 <= value <= 1 for value in pair) for pair in controls):
            raise ValueError("Controls must be nonempty [drive, mix] pairs in [0, 1]")
    known_controls = {tuple(pair) for pair in train_controls}
    count = 0
    for split in SPLITS:
        entries = []
        for index in range(settings["sources"][split]):
            source = f"{split}_source_{index:04d}"
            t = np.arange(samples) / sample_rate
            fundamental = rng.uniform(65, 700)
            # Broadband transients + independently sampled harmonic performances.
            dry = sum(rng.uniform(0.04, 0.16) * np.sin(2 * np.pi * fundamental * k * t + rng.uniform(-np.pi, np.pi)) for k in range(1, 5))
            dry *= (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 8) * t) ** 2)
            dry += rng.normal(0, 0.025, samples)
            dry *= rng.uniform(0.3, 1.0)
            dry = dry.astype(np.float32)
            controls = control_sets[split]
            for preset, values in enumerate(controls):
                pair_id = f"{source}_preset_{preset}"
                pair_dir = root / "pairs" / pair_id
                pair_dir.mkdir(parents=True)
                wet = target(dry, values, settings["effect"], sample_rate)
                wavfile.write(pair_dir / "dry.wav", sample_rate, dry)
                wavfile.write(pair_dir / "wet.wav", sample_rate, wet)
                entry = {
                    "pair_id": pair_id, "source_id": source, "split": split,
                    "regime": "seen_controls" if tuple(values) in known_controls else "held_out_controls",
                    "effect_type": settings["effect"], "device_id": "synthetic_dsp_v1",
                    "sample_rate": sample_rate, "channels": 1, "num_samples": samples,
                    "latency_samples": 0, "input_gain_db": 0.0,
                    "parameters": dict(zip(("drive", "mix"), values)),
                    "source_license": "CC0-1.0", "provenance": "procedurally generated; no external recordings",
                    "capture": {"method": "software", "version": "1"},
                    "dry": f"pairs/{pair_id}/dry.wav", "wet": f"pairs/{pair_id}/wet.wav",
                    "dry_sha256": sha256(pair_dir / "dry.wav"),
                    "wet_sha256": sha256(pair_dir / "wet.wav"),
                }
                write_json(pair_dir / "params.json", entry)
                entries.append(entry)
                count += 1
        (root / "manifests").mkdir(exist_ok=True)
        (root / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    write_json(root / "dataset.json", {"seed": data_seed, "data": settings,
                                        "generation_hash": canonical_hash({"seed": data_seed, "data": settings}), "pairs": count})
    return verify(root)


def verify(root):
    root = Path(root).resolve()
    source_splits, content_splits, pair_ids = {}, {}, set()
    sample_rates, effects = set(), set()
    counts = {}
    for split in SPLITS:
        entries = records(root, split)
        if not entries:
            raise ValueError(f"Empty split: {split}")
        counts[split] = len(entries)
        for entry in entries:
            for field in ("pair_id", "source_id", "split", "effect_type", "device_id", "sample_rate", "channels", "num_samples", "latency_samples", "parameters", "source_license", "provenance"):
                if field not in entry:
                    raise ValueError(f"Missing field: {field}")
            if entry["pair_id"] in pair_ids or entry["split"] != split:
                raise ValueError("Duplicate pair ID or incorrect split")
            pair_ids.add(entry["pair_id"])
            if entry["channels"] != 1 or not entry["source_license"] or not entry["provenance"]:
                raise ValueError("Mono audio and provenance/license required")
            source = entry["source_id"]
            if source_splits.setdefault(source, split) != split:
                raise ValueError(f"Source leakage across splits: {source}")
            for side in ("dry", "wet"):
                path = (root / entry[side]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Manifest path escapes dataset root")
                digest = sha256(path)
                if digest != entry[f"{side}_sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
                if side == "dry" and content_splits.setdefault(digest, split) != split:
                    raise ValueError("Identical dry content appears in multiple splits")
            dry, wet = aligned_pair(root, entry)
            if len(dry) != entry["num_samples"]:
                raise ValueError("Aligned length disagrees with num_samples")
            if max(np.abs(dry).max(), np.abs(wet).max()) > 1:
                raise ValueError("Audio outside full scale; adjust capture gain without independent normalization")
            for key in ("drive", "mix"):
                value = entry["parameters"][key]
                if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid normalized control: {key}")
            sample_rates.add(entry["sample_rate"])
            effects.add(entry["effect_type"])
    if len(sample_rates) != 1 or len(effects) != 1:
        raise ValueError("Each experiment requires one sample rate and one effect family")
    return {"passed": True, "pairs": counts, "sources": len(source_splits),
            "sample_rate": next(iter(sample_rates)), "effect": next(iter(effects)),
            "manifest_sha256": {split: sha256(root / "manifests" / f"{split}.jsonl") for split in SPLITS}}


class PairedDataset(Dataset):
    def __init__(self, root, split, segment_samples=None, context_samples=0):
        self.root = Path(root)
        self.entries = records(root, split)
        self.segment_samples = segment_samples
        self.context_samples = context_samples
        self.cache = [aligned_pair(root, entry) for entry in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dry, wet = self.cache[index]
        if self.segment_samples is not None:
            segment = self.segment_samples
            if len(dry) < segment:
                raise ValueError("Recording is shorter than training segment")
            start = int(torch.randint(0, len(dry) - segment + 1, ()).item())
            # Real preceding context, with zero padding only before source start.
            left = max(0, start - self.context_samples)
            x = np.pad(dry[left:start + segment], (max(0, self.context_samples - start), 0))
            y = wet[start:start + segment]
        else:
            x, y = dry, wet
        controls = self.entries[index]["parameters"]
        return (torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None],
                torch.tensor([controls["drive"], controls["mix"]], dtype=torch.float32))
