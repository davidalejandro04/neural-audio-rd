# Neural Audio R&D

A runnable, phased research pipeline for controllable neural audio effects, based on [the HTML research report](deep-research-report.html) and [the Markdown report](deep-research-report.md).

The [research production plan](docs/RESEARCH_PIPELINE.md) defines seven phases, measurable acceptance criteria, current implementation status and the remaining work. The first implementation supports synthetic distortion/cabinet/reverb pairs, data audits, causal Conv1D/TCN/GRU baselines, FiLM conditioning, a TCN/GRU/FIR hybrid with branch ablations, held-out evaluation, streaming checks, CPU measurements, ONNX export and WAV rendering. Reference transfer and a native VST3 release are future phases.

## Quickstart

Python 3.11+; run from the repository root. Install PyTorch appropriate for your machine first if needed. The reference pipeline uses CPU inference/training.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,export]"
python -m pytest -q
python -m neural_fx run --config configs/smoke.json --output runs/smoke --export
```

To benchmark all seven architectures:

```powershell
python -m neural_fx run --config configs/smoke.json --output runs/suite --suite
```

Use a new output directory for each run. `--export` requires the export extra and currently supports models without a GRU. A suite can finish with exit code 1 because some models miss the declared quality or CPU budget; inspect `status.json` and `models/*/gates.json`. Tests and engineering acceptance gates serve different purposes.

If Python reports `No module named neural_fx`, install the package from the repository root using the same interpreter you use to run it:

```console
python -m pip install -e ".[dev,export]"
python -m neural_fx --help
```

The source package lives under `src/`; changing into the repository alone does not install it. Repeat installation when switching Python interpreters or virtual environments.

Without installation, set `$env:PYTHONPATH = 'src'` on PowerShell or `PYTHONPATH=src` on POSIX. In a restricted environment, direct pytest temporary files into the repository with `--basetemp runs/pytest-local` (pytest clears that directory on reuse; use it only for disposable test output).

## Individual stages

```powershell
python -m neural_fx generate --config configs/smoke.json --output data/processed/toy
python -m neural_fx verify --data data/processed/toy
python -m neural_fx train --config configs/smoke.json --data data/processed/toy --output runs/tcn
python -m neural_fx evaluate --checkpoint runs/tcn/best.pt --data data/processed/toy --output runs/tcn/evaluation
python -m neural_fx benchmark --checkpoint runs/tcn/best.pt --output runs/tcn/runtime.json
python -m neural_fx export --checkpoint runs/tcn/best.pt --output exports/tcn.onnx
python -m neural_fx render --checkpoint runs/tcn/best.pt --input path/to/mono-48k.wav --output runs/rendered.wav --drive 0.5 --mix 1
```

Change `data.effect` to `cabinet` or `reverb` in a copied config to generate deterministic IR targets. The short hybrid FIR is a prototype; long reverb modeling needs the additional work specified in the plan.

## Listening and own captures

The [data contract](data/README.md) explains the explicit capture index, signed latency and source splits. `prepare` imports it without changing signal levels.

```powershell
python -m neural_fx abx-package --evaluation runs/tcn/evaluation --output runs/listening
python -m neural_fx abx-score --answers runs/listening/private/answers.json --responses path/to/completed-responses.csv --output runs/listening/scores.json
```

Share only the `public` folder with listeners. Keep the private key with the researcher. The scorer reports per-listener descriptive binomial tests and confidence intervals; a confirmatory group study needs the protocol in the phase plan.

## Evidence and limitations

See [the initial results](reports/INITIAL_RESULTS.md) for measured results and failed gates. Data, weights, exports and audio remain local under ignored directories; compact metric evidence is kept in `reports/`. External recordings are not bundled. Code, recordings, trained weights and plugin dependencies require separate license decisions before distribution.

The implementation does not establish architectural novelty, real hardware fidelity, reference-based effect transfer, listening-test indistinguishability or plugin readiness. Those claims require the remaining phase gates.
