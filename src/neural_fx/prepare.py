"""Import an explicit, license-bearing capture index without changing levels."""
import json
import re
from pathlib import Path

from scipy.io import wavfile

from .data import SPLITS, aligned_pair, verify
from .utils import sha256, write_json


def prepare_pairs(index, output):
    index, output = Path(index).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset already exists: {output}")
    entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError("Capture index is empty")
    split_entries = {split: [] for split in SPLITS}
    seen = set()
    for entry in entries:
        for key in ("pair_id", "source_id", "split", "dry", "wet", "sample_rate", "latency_samples", "parameters", "effect_type", "device_id", "source_license", "provenance"):
            if key not in entry:
                raise ValueError(f"Capture index missing {key}")
        identifier = entry["pair_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("pair_id must be unique and contain only letters, digits, underscores or hyphens")
        seen.add(identifier)
        if entry["split"] not in SPLITS:
            raise ValueError("split must be train, valid or test; assign by source recording")
        for side in ("dry", "wet"):
            if not (index.parent / entry[side]).resolve().is_relative_to(index.parent):
                raise ValueError("Capture paths must stay within the index directory")
        dry, wet = aligned_pair(index.parent, entry)
        destination = output / "pairs" / identifier
        destination.mkdir(parents=True)
        wavfile.write(destination / "dry.wav", entry["sample_rate"], dry)
        wavfile.write(destination / "wet.wav", entry["sample_rate"], wet)
        result = {**entry, "original_latency_samples": entry["latency_samples"],
                  "latency_samples": 0, "num_samples": len(dry), "channels": 1,
                  "source_files": {side: {"path": entry[side], "sha256": sha256(index.parent / entry[side])} for side in ("dry", "wet")}}
        for side in ("dry", "wet"):
            result[side] = f"pairs/{identifier}/{side}.wav"
            result[f"{side}_sha256"] = sha256(destination / f"{side}.wav")
        write_json(destination / "params.json", result)
        split_entries[entry["split"]].append(result)
    (output / "manifests").mkdir()
    for split, selected in split_entries.items():
        (output / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in selected), encoding="utf-8")
    result = verify(output)
    write_json(output / "capture_index.json", {"index_sha256": sha256(index), "pairs": len(entries)})
    return result
