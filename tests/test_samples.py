import numpy as np
import pytest

from neural_fx.data import generate, read_audio, records
from neural_fx.dsp import distortion
from neural_fx.samples import preview_samples
from neural_fx.utils import read_json, sha256


def test_audition_pack_preserves_pairs_and_records_sequence_transforms(tmp_path):
    config = read_json("configs/smoke.json")
    config["data"].update({"samples": 512, "sources": {"train": 1, "valid": 1, "test": 2}})
    dataset = tmp_path / "dataset"
    generate(config, dataset)
    output = tmp_path / "samples"
    result = preview_samples(dataset, output)
    assert result["clips"] == 6
    manifest = read_json(output / "manifest.json")
    source = {r["pair_id"]: r for r in records(dataset, "test")}
    for clip in manifest["clips"]:
        for kind in ("dry", "wet"):
            path = output / clip[kind]
            assert sha256(path) == manifest["files_sha256"][path.name]
            rate, samples = read_audio(path)
            assert rate == clip["sample_rate"] and np.isfinite(samples).all()
            if clip["transform"] == "byte-identical dataset pair":
                assert path.read_bytes() == (dataset / source[clip["source_pairs"][0]][kind]).read_bytes()
            else:
                assert len(samples) == len(clip["source_pairs"]) * (512 + int(rate * .1))
        if clip["transform"] != "byte-identical dataset pair":
            _, dry = read_audio(output / clip["dry"])
            _, wet = read_audio(output / clip["wet"])
            np.testing.assert_array_equal(wet, distortion(dry, clip["drive"], clip["mix"]))
    with pytest.raises(FileExistsError):
        preview_samples(dataset, output)
