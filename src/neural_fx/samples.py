"""Small, traceable audition packs made from existing synthetic test pairs."""
import shutil
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .data import read_audio, records
from .dsp import distortion
from .utils import sha256, write_json


def preview_samples(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Sample output already exists: {output}")
    rows = [r for r in records(dataset, "test") if r["effect_type"] == "distortion"]
    if not rows or any(r.get("capture", {}).get("method") != "software" for r in rows):
        raise ValueError("Preview samples require synthetic distortion test pairs")
    groups = {}
    for row in rows:
        key = tuple(row["parameters"][p] for p in ("drive", "mix"))
        if key[1] == 1:
            groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("Need fully wet distortion pairs")
    selected = sorted(groups)[:2]
    output.mkdir(parents=True, exist_ok=True)
    clips = []
    for group_index, controls in enumerate(selected):
        group = groups[controls]
        for index, row in enumerate(group[:2]):
            stem = f"source-{group_index + 1}-{index + 1}"
            for kind in ("dry", "wet"):
                source = dataset / row[kind]
                if sha256(source) != row[f"{kind}_sha256"]:
                    raise ValueError(f"Dataset checksum mismatch: {source}")
                shutil.copyfile(source, output / f"{stem}-{kind}.wav")
            clips.append({"name": f"Source {group_index * 2 + index + 1:02d} / Drive {controls[0]:.0%}",
                          "dry": f"{stem}-dry.wav", "wet": f"{stem}-wet.wav",
                          "drive": controls[0], "mix": controls[1], "sample_rate": row["sample_rate"],
                          "source_pairs": [row["pair_id"]], "transform": "byte-identical dataset pair"})
        # Longer audition clips retain a fixed target setting across disjoint source snippets.
        parts = {"dry": [], "wet": []}
        rate = group[0]["sample_rate"]
        for row in group[:12]:
            for kind in parts:
                source = dataset / row[kind]
                if sha256(source) != row[f"{kind}_sha256"]:
                    raise ValueError(f"Dataset checksum mismatch: {source}")
            actual_rate, audio = read_audio(dataset / row["dry"])
            if actual_rate != rate:
                raise ValueError("Sample sequence requires matching sample rates")
            fade = min(int(rate * .005), len(audio) // 2)
            audio = audio.copy()
            audio[:fade] *= np.linspace(0, 1, fade)
            audio[-fade:] *= np.linspace(1, 0, fade)
            silence = np.zeros(int(rate * .1), dtype=np.float32)
            parts["dry"].extend((audio, silence))
            parts["wet"].extend((distortion(audio, *controls), silence))
        stem = f"sequence-{group_index + 1}"
        for kind in parts:
            wavfile.write(output / f"{stem}-{kind}.wav", rate, np.concatenate(parts[kind]))
        clips.insert(group_index, {"name": f"Dataset sequence / Drive {controls[0]:.0%}",
                     "dry": f"{stem}-dry.wav", "wet": f"{stem}-wet.wav", "drive": controls[0],
                     "mix": controls[1], "sample_rate": rate,
                     "source_pairs": [r["pair_id"] for r in group[:12]],
                     "transform": "concatenated dry sources; 5 ms edge fades; 100 ms gaps; wet recomputed with known DSP"})
    manifest = {"scope": "Synthetic distortion audition samples, not hardware recordings or new evaluation evidence",
                "provenance": "Existing procedurally generated test split; CC0-1.0 source audio",
                "source_manifest_sha256": sha256(dataset / "manifests/test.jsonl"), "clips": clips,
                "files_sha256": {p.name: sha256(p) for p in sorted(output.glob("*.wav"))}}
    write_json(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        "Synthetic distortion audition pack\n\nSelect a clip in NeuralAudioFX and press Play. Loop is enabled by default.\n"
        "The player supplies dry audio to the loaded model; Dry compare auditions the delayed dry signal.\n"
        "Selecting a dataset clip restores its matching Drive/Mix settings.\n"
        "*-wet.wav files are known-DSP targets for offline comparison, not neural renders.\n"
        "Sequences use 5 ms dry edge fades and 100 ms gaps, with DSP targets recomputed after fading. Source pairs are byte-identical copies.\n"
        "Original clips are short research fixtures. manifest.json preserves hashes, source IDs and transforms.\n"
        "Source audio is procedurally generated, CC0-1.0; no external recordings.\n", encoding="utf-8")
    return {"samples": str(output), "clips": len(clips)}
