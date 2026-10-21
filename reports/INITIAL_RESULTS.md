# Initial implementation results

Measured on 2026-10-20 using `configs/smoke.json`: seed 1337, 48 kHz mono, 2048 samples per source, 8/3/3 train/validation/test sources, 62 pairs, 12 CPU epochs per model. These are pipeline checks on synthetic data, not publication results. Original research reports are preserved unchanged.

| Model | Parameters | Test ESR | MR-STFT | Max stream error | Worst p99 / deadline | Engineering gates |
|---|---:|---:|---:|---:|---:|---|
| conditional_tcn | 769 | 0.2232 | 1.2587 | 0 | 0.271 | Pass |
| conv1d | 225 | 0.2051 | 1.1207 | 0 | 0.088 | Pass |
| gru | 321 | 0.4069 | 1.7393 | 3e-08 | 0.882 | Fail: CPU budget |
| hybrid | 1243 | 0.2651 | 1.4748 | 1.5e-08 | 1.306 | Fail: CPU budget |
| hybrid_no_fir | 1096 | 0.3442 | 1.6332 | 1.5e-08 | 1.196 | Fail: CPU budget |
| hybrid_no_gru | 919 | 0.2629 | 1.3575 | 0 | 0.328 | Pass |
| tcn | 625 | 0.1951 | 1.0700 | 0 | 0.212 | Pass |

ESR and MR-STFT average active presets within each source, then take the median across test sources. Bypass is excluded. All seven models beat identity at the declared smoke threshold and satisfy ESR <=0.5. GRU-containing models miss the p99 <=70% callback budget in the isolated Python CPU screening. These runtime reports use 20 warmup and 200 timed blocks per size, not the ten-minute native acceptance test. Timing varies between runs.

The simpler TCN performs best in this small experiment. This does not test the hybrid hypothesis: the models are not parameter matched, the dataset has one memoryless effect family, and only one seed was run. Do not infer hybrid superiority or hardware fidelity.

The conditional TCN completed a separate end-to-end run including ONNX export. Maximum ONNX/PyTorch error was 1.34e-07 against a 1e-5 tolerance at block size 128. A WAV render and a three-trial ABX package were generated from the checkpoint. No human listening responses were collected.

Verification: 38 tests passed, including deterministic data/training, alignment, leakage/checksum rejection, convolution, exact bypass, causality, regular/irregular streaming, state reset, gate rejection, ABX scoring and ONNX sequence parity. Docker and GitHub Actions definitions were added; Docker execution was unavailable because its daemon was not running. CI has not been executed remotely.

Local artifacts: `runs/initial/models/*/best.pt`, `runs/verified/models/conditional_tcn/model.onnx`, `runs/verified/rendered.wav`, and `runs/verified/abx/`. They remain ignored by Git. [Machine-readable evidence](smoke-evidence.json) retains config, dataset hashes, per-model metrics/history/runtime/gates, environment and export identity. [Production phase plan](../docs/RESEARCH_PIPELINE.md) describes the remaining work.
