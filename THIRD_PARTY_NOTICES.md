# Local native development dependencies

The implementation links external projects rather than copying their processing implementations into `src/` or `deploy/`.

| Component | Source used | Local location |
|---|---|---|
| ONNX Runtime CPU | Official Windows x64 SDK 1.23.2 | `.deps/native/onnxruntime-win-x64-1.23.2` |
| JUCE | Official source tag 8.0.12 | `.deps/native/JUCE-8.0.12` |
| Steinberg VST3 validator | SDK commit `3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96` with pinned submodules | `.deps/native/vst3sdk` |

`native-build` verifies SHA-256 values of downloaded ORT/JUCE archives. `validator-build` pins the Steinberg source revision. Build manifests and validation reports record executable and model identities. The runtime DLL is named `neural_fx_ort_123.dll` locally to isolate it from a different `onnxruntime.dll` supplied by Windows or another host. Its binary contents are the official SDK DLL.

The local preview copies dependency license/notice files into its `licenses/` folder. Consult the complete original dependency sources for notices associated with their bundled components, including JUCE's VST3 SDK snapshot. The preview is a local development artifact; its creation does not choose a distribution license for the project's code or model weights.

Original dependency sources: [ONNX Runtime](https://github.com/microsoft/onnxruntime), [JUCE](https://github.com/juce-framework/JUCE), [VST3 SDK](https://github.com/steinbergmedia/vst3sdk). No external research weights are included.

## GuitarJam audition recordings

Three dry electric-guitar excerpts from [Julian-br/GuitarJam](https://huggingface.co/datasets/Julian-br/GuitarJam) are included in `samples/guitar/` and the v4 preview. The publisher declares **CC0-1.0** and describes raw Fender Stratocaster DI recordings without effects or amplifiers. Source files `guitar_jam/test/026.wav`, `050.wav` and `085.wav` are pinned to revision `2d467bfec90af19301b01123494f4b2ba64c5a3a`.

The local excerpts use the first ten seconds, polyphase resampling from 44.1 to 48 kHz and 5 ms edge fades, preserving the source level. Original checksums, URLs and transformations are in `configs/guitarjam_samples.json` and `samples/guitar/manifest.json`. The original publisher's declaration is retained in `samples/guitar/SOURCE_DATASET_CARD.md`; see `samples/guitar/LICENSE.md` and the [CC0 legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode). Credit is retained for provenance. No paired hardware wet targets are supplied.
