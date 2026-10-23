import json

import numpy as np
import pytest
from scipy.io import wavfile

from neural_fx.data import read_audio
from neural_fx.guitar_samples import guitar_samples
from neural_fx.utils import read_json, sha256


def test_cc0_guitar_import_resamples_without_normalizing_and_preserves_provenance(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    source = cache / "026.wav"
    x = (np.sin(np.arange(44100) * 2 * np.pi * 220 / 44100) * 1000).astype(np.int16)
    wavfile.write(source, 44100, x)
    card = cache / "DATASET_CARD.md"
    card.write_text("---\nlicense: cc0-1.0\n---\nFixture", encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"dataset": "fixture", "source_page": "https://example.org", "revision": "pinned",
                               "license": "CC0-1.0", "retrieved_date": "2026-10-22", "dataset_card_sha256": sha256(card),
                               "files": [{"path": "test/026.wav", "url": "https://example.org/026.wav", "sha256": sha256(source)}]}))
    output = tmp_path / "samples"
    result = guitar_samples(spec, output, cache, seconds=1)
    assert result["clips"] == 1
    manifest = read_json(output / "manifest.json")
    clip = manifest["clips"][0]
    rate, y = read_audio(output / clip["dry"])
    assert rate == 48000 and len(y) == 48000
    assert .029 < np.max(np.abs(y)) < .032
    assert y[0] == 0 and y[-1] == 0
    assert "wet" not in clip and clip["source_sha256"] == sha256(source)
    assert clip["engine"] == 1
    for name, checksum in manifest["files_sha256"].items():
        assert sha256(output / name) == checksum
    source.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        guitar_samples(spec, tmp_path / "bad-samples", cache)
