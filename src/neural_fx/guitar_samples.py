"""Import pinned, CC0 electric-guitar DI excerpts for effect auditioning."""
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .data import read_audio
from .utils import read_json, sha256, write_json


def guitar_samples(spec, output, cache="data/raw/guitarjam", seconds=10):
    settings = read_json(spec)
    output, cache = Path(output), Path(cache)
    if settings["license"] != "CC0-1.0" or not 1 <= seconds <= 30:
        raise ValueError("Expected CC0 sources and 1..30 second excerpts")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Sample output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in settings["files"]:
        name = Path(row["path"]).name
        source = cache / name
        if not source.exists():
            source.write_bytes(urllib.request.urlopen(row["url"], timeout=60).read())
        if sha256(source) != row["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        rate, original = read_audio(source)
        if rate != 44100:
            raise ValueError("Pinned GuitarJam sources must use 44.1 kHz")
        audio = resample_poly(original[:int(seconds * rate)], 160, 147).astype(np.float32)
        fade = min(240, len(audio) // 2)
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
        target = output / f"guitarjam-{Path(name).stem}-dry.wav"
        wavfile.write(target, 48000, audio)
        entries.append({"name": f"Electric guitar DI / GuitarJam {Path(name).stem}", "dry": target.name,
                        "sample_rate": 48000, "drive": .65, "mix": 1., "engine": 1,
                        "inputDb": 6., "outputDb": -9., "tone": 5500.,
                        "source_url": row["url"], "source_sha256": row["sha256"], "source_path": row["path"],
                        "source_license": "CC0-1.0", "source_sample_rate": rate,
                        "original_seconds": len(original) / rate, "seconds": len(audio) / 48000,
                        "transform": f"first {seconds:g} seconds; polyphase resampling 44100->48000; 5 ms edge fades; no normalization"})
    # Original license declaration is retained separately from audio transformations.
    card = cache / "DATASET_CARD.md"
    if not card.exists():
        url = f"https://huggingface.co/datasets/{settings['dataset']}/raw/{settings['revision']}/README.md"
        card.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    if sha256(card) != settings["dataset_card_sha256"]:
        raise ValueError("Dataset card checksum mismatch")
    shutil.copyfile(card, output / "SOURCE_DATASET_CARD.md")
    (output / "LICENSE.md").write_text(
        "# GuitarJam audio license\n\nThe source publisher declares CC0-1.0 in the accompanying SOURCE_DATASET_CARD.md.\n"
        f"Source: {settings['source_page']}\nPinned revision: {settings['revision']}\n\n"
        "CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/\n"
        "Legal code: https://creativecommons.org/publicdomain/zero/1.0/legalcode\n"
        "Recording credited to Julian-br / GuitarJam. This credit is retained for provenance.\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Electric guitar DI audition clips / GuitarJam / CC0-1.0\n\n"
        "These files are dry direct-input guitar performances. No distortion is baked into them.\n"
        "Select a clip, choose Pedal distortion (4x), press Play sample, and adjust Drive.\n"
        "The supplied Crunch settings are Drive 65%, Mix 100%, Input +6 dB, Tone 5500 Hz, Output -9 dB.\n"
        "Dry compare auditions the original input with matched buffering latency; it is not loudness matched.\n"
        "Neural model remains separately selectable; Input/Tone apply to the pedal mode.\n"
        "No hardware wet ground truth is included. These are audition examples, not real-pedal validation.\n"
        "See manifest.json, SOURCE_DATASET_CARD.md and LICENSE.md for source URLs, hashes and transformations.\n", encoding="utf-8")
    manifest = {"scope": "Real CC0 electric-guitar DI audition inputs; no hardware wet targets or training split claims",
                "source_page": settings["source_page"], "revision": settings["revision"], "license": settings["license"],
                "retrieved_date": settings["retrieved_date"], "clips": entries,
                "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return {"samples": str(output), "clips": len(entries), "license": settings["license"]}
