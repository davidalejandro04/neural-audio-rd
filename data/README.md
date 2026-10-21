# Data contract

External recordings and generated WAV files remain in ignored local directories. Do not mirror research datasets or copy their license claims without verifying the actual files and terms. The reports list possible datasets, not approved dependencies.

Each prepared dataset has `manifests/train.jsonl`, `valid.jsonl`, `test.jsonl` and `pairs/<pair_id>/{dry.wav,wet.wav,params.json}`. Manifests preserve source and device IDs, source license and provenance, effect family, normalized `drive`/`mix` controls, sample rate, channel count, sample count, latency, file paths and SHA-256 hashes. Synthetic data are algorithmic signals dedicated to CC0-1.0; no external music is used.

Synthetic generation splits **source performances before rendering presets**. Test performances never occur in training/validation. Byte-identical dry WAVs in different splits are also rejected. Re-encoded duplicates still require provenance review. Training reads random paired crops with preceding context; tests evaluate complete recordings. No dry/wet peak normalization is applied. The importer only supports mono, existing matched sample rates, and the `drive`/`mix` control contract. Extend control mappings explicitly for other hardware.

For real captures, create an index JSONL beside the raw WAVs, one object per pair. Example object (repeat with unique pair IDs and at least one independent source in each split):

```json
{"pair_id":"pedal_take01_preset1","source_id":"di_take01","split":"train","dry":"take01_dry.wav","wet":"take01_wet.wav","sample_rate":48000,"latency_samples":137,"parameters":{"drive":0.5,"mix":1.0},"effect_type":"distortion","device_id":"own_pedal_A","source_license":"own-recording","provenance":"Recorded by repository owner; source performance take01"}
```

Positive latency means the wet path lags dry: the importer removes that many samples from wet. Negative latency removes samples from dry. It keeps the common valid duration, writes float32 WAVs without gain normalization, records original lag and file hashes, and sets prepared latency to zero. Supply a measured lag; this importer does not estimate latency or resample automatically. Inspect alignment before accepting a capture dataset. Keep all presets and segments of a performance under one `source_id` and split.

```powershell
python -m neural_fx prepare --index data/raw/own_captures/index.jsonl --output data/processed/own_captures
python -m neural_fx verify --data data/processed/own_captures
```

No automatic split repair is performed. Invalid data fail verification. Prepared captures have no synthetic `dataset.json`, so evaluation does not label an analytic model as their ground-truth oracle.
