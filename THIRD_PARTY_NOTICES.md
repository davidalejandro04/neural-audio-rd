# Local native development dependencies

The implementation links external projects rather than copying their processing implementations into `src/` or `deploy/`.

| Component | Source used | Local location |
|---|---|---|
| ONNX Runtime CPU | Official Windows x64 SDK 1.23.2 | `.deps/native/onnxruntime-win-x64-1.23.2` |
| JUCE | Official source tag 8.0.12 | `.deps/native/JUCE-8.0.12` |
| Steinberg VST3 validator | SDK commit `3cdf9ca5d1f5b1b21e0a86832aa4abe55607bd96` with pinned submodules | `.deps/native/vst3sdk` |

`native-build` verifies SHA-256 values of downloaded ORT/JUCE archives. `validator-build` pins the Steinberg source revision. Build manifests and validation reports record executable and model identities. The runtime DLL is named `neural_fx_ort_123.dll` locally to isolate it from a different `onnxruntime.dll` supplied by Windows or another host. Its binary contents are the official SDK DLL.

The local preview copies dependency license/notice files into its `licenses/` folder. Consult the complete original dependency sources for notices associated with their bundled components, including JUCE's VST3 SDK snapshot. The preview is a local development artifact; its creation does not choose a distribution license for the project's code or model weights.

Original sources: [ONNX Runtime](https://github.com/microsoft/onnxruntime), [JUCE](https://github.com/juce-framework/JUCE), [VST3 SDK](https://github.com/steinbergmedia/vst3sdk). No external music datasets, research weights, or third-party recordings are included.
