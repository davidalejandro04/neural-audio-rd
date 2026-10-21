import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.io import wavfile

from neural_fx.data import PairedDataset, aligned_pair, generate, records, verify
from neural_fx.utils import read_json


@pytest.fixture
def config():
    value = read_json("configs/smoke.json")
    value["data"]["sources"] = {"train": 2, "valid": 1, "test": 1}
    value["data"]["samples"] = 512
    return value


def test_deterministic_generation_and_source_splits(tmp_path, config):
    a, b = tmp_path / "a", tmp_path / "b"
    assert generate(config, a) == generate(config, b)
    split_sources = [{r["source_id"] for r in records(a, split)} for split in ("train", "valid", "test")]
    assert all(not left & right for i, left in enumerate(split_sources) for right in split_sources[i + 1:])
    data = PairedDataset(a, "train", segment_samples=256, context_samples=14)
    dry, wet, controls = data[0]
    assert dry.shape == (1, 270) and wet.shape == (1, 256) and controls.shape == (2,)
    with pytest.raises(FileExistsError):
        generate(config, a)


def test_corrupt_audio_rejected(tmp_path, config):
    generate(config, tmp_path)
    record = records(tmp_path, "train")[0]
    with (tmp_path / record["wet"]).open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path)


def test_source_leakage_rejected(tmp_path, config):
    generate(config, tmp_path)
    entries = records(tmp_path, "test")
    entries[0]["source_id"] = records(tmp_path, "train")[0]["source_id"]
    (tmp_path / "manifests/test.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
    with pytest.raises(ValueError, match="Source leakage"):
        verify(tmp_path)


@pytest.mark.parametrize("lag", [-7, 0, 7])
def test_latency_alignment(tmp_path, lag):
    rng = np.random.default_rng(42)
    signal = rng.uniform(-0.5, 0.5, 100).astype(np.float32)
    dry = np.pad(signal, (max(-lag, 0), 0))
    wet = np.pad(signal, (max(lag, 0), 0))
    wavfile.write(tmp_path / "dry.wav", 48000, dry)
    wavfile.write(tmp_path / "wet.wav", 48000, wet)
    x, y = aligned_pair(tmp_path, {"dry": "dry.wav", "wet": "wet.wav", "sample_rate": 48000, "latency_samples": lag})
    np.testing.assert_array_equal(x, y)


def test_mismatched_rate_rejected(tmp_path):
    signal = np.zeros(100, dtype=np.float32)
    wavfile.write(tmp_path / "dry.wav", 48000, signal)
    wavfile.write(tmp_path / "wet.wav", 44100, signal)
    with pytest.raises(ValueError, match="Sample-rate mismatch"):
        aligned_pair(tmp_path, {"pair_id": "a", "dry": "dry.wav", "wet": "wet.wav", "sample_rate": 48000, "latency_samples": 0})
