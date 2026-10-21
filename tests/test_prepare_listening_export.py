import csv
import importlib.util
import json

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from neural_fx.data import read_audio, records
from neural_fx.listening import package_abx, score_abx
from neural_fx.models import AudioModel
from neural_fx.prepare import prepare_pairs
from neural_fx.runtime import export_onnx
from neural_fx.utils import read_json


def test_capture_import_preserves_level_and_alignment(tmp_path):
    index = tmp_path / "index.jsonl"
    rng = np.random.default_rng(1)
    entries = []
    for split in ("train", "valid", "test"):
        x = rng.uniform(-0.3, 0.3, 512).astype(np.float32)
        wet = np.pad(x * 0.5, (7, 0))
        wavfile.write(tmp_path / f"{split}_dry.wav", 48000, x)
        wavfile.write(tmp_path / f"{split}_wet.wav", 48000, wet)
        entries.append({"pair_id": split, "source_id": split, "split": split,
                        "dry": f"{split}_dry.wav", "wet": f"{split}_wet.wav", "sample_rate": 48000,
                        "latency_samples": 7, "parameters": {"drive": 0.5, "mix": 1},
                        "effect_type": "distortion", "device_id": "test", "source_license": "own-recording", "provenance": "test fixture"})
    index.write_text("\n".join(json.dumps(e) for e in entries))
    output = tmp_path / "prepared"
    assert prepare_pairs(index, output)["passed"]
    entry = records(output, "train")[0]
    assert entry["latency_samples"] == 0 and entry["original_latency_samples"] == 7
    _, dry = read_audio(output / entry["dry"])
    _, wet = read_audio(output / entry["wet"])
    np.testing.assert_allclose(wet, dry * 0.5, atol=0)
    assert not (output / "dataset.json").exists()


def test_unsigned_pcm_read_scale(tmp_path):
    wavfile.write(tmp_path / "u8.wav", 48000, np.array([0, 128, 255], dtype=np.uint8))
    _, values = read_audio(tmp_path / "u8.wav")
    np.testing.assert_array_equal(values, [-1, 0, 127 / 128])


def test_abx_package_and_exact_scoring(tmp_path):
    example = tmp_path / "evaluation" / "audio" / "pair"
    example.mkdir(parents=True)
    x = np.linspace(-0.2, 0.2, 512, dtype=np.float32)
    wavfile.write(example / "target.wav", 48000, x)
    wavfile.write(example / "model.wav", 48000, x * 0.5)
    output = tmp_path / "abx"
    assert package_abx(tmp_path / "evaluation", output)["trials"] == 1
    private = read_json(output / "private/answers.json")
    trial = private["trials"][0]
    assert (output / "public/trial_001/X.wav").read_bytes() == (output / f"public/trial_001/{trial['answer']}.wav").read_bytes()
    assert not (output / "public/answers.json").exists()
    responses = tmp_path / "responses.csv"
    with responses.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["listener_id", "trial_id", "choice"])
        writer.writerow(["anonymous_1", trial["trial_id"], trial["answer"]])
    score = score_abx(output / "private/answers.json", responses, tmp_path / "score.json")
    assert score["listeners"][0]["accuracy"] == 1
    assert score["listeners"][0]["binomial_p"] == 1


@pytest.mark.skipif(any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")), reason="Install the export extra")
def test_stateful_onnx_export_parity(tmp_path):
    config = read_json("configs/smoke.json")
    model = AudioModel(**config["model"])
    checkpoint = tmp_path / "test.pt"
    torch.save({"model_spec": config["model"], "config": config, "state_dict": model.state_dict()}, checkpoint)
    exported = export_onnx(checkpoint, tmp_path / "test.onnx", 64)
    assert exported["passed"] and exported["max_error"] < 1e-5
    golden = np.load(tmp_path / "test.golden.npz")
    assert golden["audio"].shape == golden["expected"].shape == (12, 1, 1, 64)


def test_gru_export_rejects_hidden_state_reset(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = read_json("configs/smoke.json")
    config["model"]["name"] = "gru"
    model = AudioModel(**config["model"])
    path = tmp_path / "gru.pt"
    torch.save({"model_spec": config["model"], "config": config, "state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="hidden-state"):
        export_onnx(path, tmp_path / "gru.onnx")
