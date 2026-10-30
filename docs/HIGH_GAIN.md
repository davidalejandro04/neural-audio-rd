# British high-gain guitar rig

The v5 native preview adds a complete guitar chain:

`DI input → selected preamp → three-band EQ → cabinet → delay → room reverb → Master`

![Sunset lead in the native editor](assets/hard-rock-rig.png)

Run `exports\preview-v5\Launch-NeuralFX.cmd`, choose a **48 kHz** audio device, select a dry guitar recording and press **Play sample**. Stop returns to live input. The launcher opens the British DSP amp with the Hard rock settings and loads the new high-gain neural model for comparison. The build stays local; it is not installed into a DAW.

## Playing the rig

| Preset | Starting point |
| --- | --- |
| Clean | Low-drive pedal, reduced input, warm vintage cabinet, small room |
| Crunch | Cascaded British DSP amp, moderate drive, focused 4x12 |
| Hard rock | 72% drive, boosted input, forward mids, focused 4x12 |
| Sunset lead | 88% drive, 340 ms echo, 32% feedback, 12% room |
| Neural lead | Currently loaded model at 85% drive, the same lead cabinet/space settings |

The musical aim is a mid-forward, sustaining late-1980s hard-rock rig, inspired by the requested Marshall / Guns N' Roses direction. These are original synthetic voicings, not a measured Marshall circuit, a Slash preset, or an artist/hardware capture. The separate gain and EQ controls follow familiar guitar-amp practice; Marshall's [JCM800 control list](https://www.marshall.com/gb/en/product/jcm800-2203-vintage-reissue-head) provides a reference for that interface vocabulary, not an emulation specification.

- **Engine:** the neural model, the existing 4x oversampled pedal, or the new 4x oversampled British amp. The DSP amp tightens low frequencies before two asymmetric saturation stages. Its implementation differs from the training target.
- **Input / DSP tone:** act only on the DSP engines. Disabled in neural mode to preserve the trained model's input/control contract.
- **Drive / Amp mix:** normalized model controls in `[drive, mix]` order. The neural model applies mix once. Intermediate mix blends the amp output before the rack; zero bypasses the rack. Master still applies at mix zero.
- **Bass / Middle / Treble:** ±12 dB shelves at 180 Hz and 3.2 kHz, and a broad 850 Hz mid band. They follow either preamp.
- **Cabinet:** Off, British, Vintage, or Modern 4x12. Each uses an 80 Hz high-pass, a broad mid emphasis, and a fourth-order high-frequency rolloff. British cuts at 5.2 kHz; Vintage at 4.2 kHz; Modern at 6.5 kHz. These are algorithmic speaker voicings, not convolution with measured IRs.
- **Delay:** 40–1000 ms, up to 65% feedback, filtered repeats and an additive level control. Time changes are smoothed and can create a pitch glide. No tempo sync or tape wow/flutter is implemented.
- **Room:** JUCE's algorithmic [Reverb](https://docs.juce.com/master/classjuce_1_1Reverb.html), with level and size. Channels have independent state and identical settings; the rig preserves mono compatibility rather than synthesizing stereo width.
- **Master:** smoothed output attenuation after the complete chain. Watch the output meter; this is not a limiter.

Changing the audition recording preserves the rig. An active preset highlights until a parameter changes. Double-click knobs to reset. Dry compare and host bypass use a continuously maintained dry delay, so neither an old neural block nor an effect tail leaks into the comparison. Effect state resets when returning from bypass.

## Trained model usage

![High-gain neural inference and post effects](assets/high-gain-neural.png)

The high-gain export is a 5,329-parameter conditional TCN with 126 samples of input history. Its new opt-in zero-preserving path subtracts the network's response to zero input under the same controls. Silence with zero history therefore remains exactly zero, including after a drive change. Historical checkpoints retain their original architecture and behavior.

The synthetic training target uses a 17-tap low-frequency tightening filter, two cascaded saturators, and a 33-tap 7 kHz low-pass. Its first-stage gain spans 6–36 dB. It contains **no cabinet, delay, or reverb**, so those effects are added separately in the native rack. It is a causal base-rate target; the oversampled DSP amp and the learned preamp are distinct signal paths. High-frequency alias suppression and hardware fidelity have not been established.

Training uses 32 procedural plucked sources, eight validation sources, and 12 held-out test sources. No GuitarJam recordings are used for training. The tracked [experiment configuration](../experiments/high-gain.json) fixes the seeds, split controls, 120 epochs, and acceptance thresholds. Validation selects the checkpoint; the test split is evaluated afterward. Exact mix-zero bypass and the affine mix law are structural.

The exporter embeds effect, sample rate, control order, mix contract, and checkpoint hash. The native engine validates explicit control/mix metadata, maintains independent audio/hidden state per channel, and reports one ONNX block of host latency (128 samples / 2.67 ms here). Legacy exports without effect metadata remain compatible and display an unspecified-effect label. A failed selection clears the active model and passes delayed dry audio; it cannot silently continue playing the old model. Rate mismatches also reject the selection. Loading a model selects neural mode. Reference presets clear post effects so their inferred Drive/Mix can be auditioned in isolation.

## Reproduce locally

```powershell
$env:PYTHONPATH = 'src;.deps'
python -m neural_fx run --config experiments/high-gain.json --output runs/my-high-gain --export
cmake --build build/native --config Release --parallel 4
python -m neural_fx native-validate --checkpoint runs/my-high-gain/models/conditional_tcn/best.pt --output runs/my-high-gain-native --block-sizes 128 --plugin build/native/plugin/NeuralAudioFX_artefacts/Release/VST3/NeuralAudioFX.vst3 --host-check build/native/plugin/nafx-plugin-check_artefacts/Release/nafx-plugin-check.exe
python -m neural_fx package-preview --validation runs/my-high-gain-native --output exports/my-high-gain --samples samples/guitar
python -m pytest -q --basetemp runs/pytest-high-gain
```

The build assumes the existing local JUCE/ONNX Runtime CMake configuration; see the [native setup guide](NATIVE_AND_REFERENCE.html) for initial setup. Use fresh run/output folders. No GitHub Actions are used.

The native `nafx-ui-check` harness accepts `NEURAL_FX_MODEL`, `NEURAL_FX_SAMPLES`, `NEURAL_FX_EFFECT_CHECK` (output folder), `NEURAL_FX_SCREENSHOT` (PNG), and `NEURAL_FX_SCREENSHOT_PRESET` (`presetLead` or `presetNeural`). Screenshots here are direct component captures from the running native editor, with actual sample playback and measured meters. The harness verifies audition transport, rejected model selection, bypass transitions, cabinets changing the signal, delay/reverb tails, mono at 44.1/96 kHz, block-size invariance, state restore, and C++ callback allocations. The allocation probe excludes external runtime internals and direct `malloc`.

See [recorded high-gain evidence](../reports/high-gain-evidence.json) for actual training, parity, timing, and rack results. These are engineering checks; target-DAW/device acceptance, human listening, hardware captures, and sustained scheduling remain unproven for v5. Earlier failed research and sustained-timing results remain unchanged.

Measured on 2026-10-29:

| Check | Result |
| --- | --- |
| Held-out synthetic source-median ESR | 0.01630; quality gate passed |
| Offline vs streamed PyTorch maximum error | 0.00000519 |
| Native / VST3 vs PyTorch maximum error | 0.000000894 |
| VST3 model controls | 15 cases: five host block sizes × wet / partial mix / dry |
| Native mono ONNX p99, 128 samples / 48 kHz | 0.187 ms; unpaced screening passed |
| Original Python timing gate | **Failed**; worst p99/deadline ratio 1.704, threshold 0.7 |
| Trained ONNX silence, including control changes | Exactly zero |
| Full-rig guitar render peaks | 0.541–0.653; no full-scale clipping on the three tested DIs |
| Real DI against synthetic target, first second | ESR 0.000518–0.05123 across six trials; not real-amp validation |
| Official VST3 validator | 47/47 unloaded and 47/47 with the high-gain model |

The initial 80-epoch high-gain trial had ESR 0.07112 and a nonzero silent-input baseline. The final 120-epoch, zero-preserving revision improves this development result, but architecture and training duration both changed; this is not a controlled architectural comparison.

Six full-rig WAVs are available locally in `runs/rig-final-checks/listening/`, with two seconds of tail each. To regenerate the compact package and audio evidence after running the native harness:

```powershell
python tools/validate_high_gain_preview.py --run runs/high-gain-v2 --native runs/high-gain-native-final --checks runs/rig-final-checks --preview exports/preview-v5 --validator runs/high-gain-validator --output reports/high-gain-evidence.json
```
