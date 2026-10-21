# Prototype model card

**Purpose:** research infrastructure and synthetic system-identification experiments. Not a validated audio plugin or a reference-conditioned transfer model.

**Architectures:** shallow causal Conv1D; unconditioned causal TCN; static FiLM-conditioned causal TCN; GRU; hybrid TCN/GRU/learned four-FIR-basis mixer; hybrid without GRU or without FIR. All take mono audio and normalized `[drive, mix]`. Exact mix=0 bypass is structural. Experiments train one effect family at a time; no device/effect embedding is implemented.

**Training:** procedurally synthesized dry audio; normalized tanh wet target in the executed benchmark. The 12-epoch CPU smoke run uses eight training, three validation and three held-out test sources. Best checkpoint is selected on validation ESR. No real recordings are included. The hybrid is the HTML report's architectural hypothesis implemented as a small prototype; its FIR path is too short for realistic reverberation.

**Evidence:** [initial results](reports/INITIAL_RESULTS.md), [recorded measurements](reports/smoke-evidence.json). Training history, checkpoints, test renders and exports are local in `runs/` and `exports/`. Artifact SHA-256 values identify local files. There was no existing Git commit at the start; later runs additionally record Python source hashes. A hash identifies an artifact, not a claim that it is publicly hosted.

**Runtime:** explicit per-stream audio history and GRU hidden state. Reset both at stream start or sample-rate changes. No lookahead; finite context is replayed per block. FiLM/router controls are constant per block, and history is recomputed under the current controls. Fast control automation, smoothing, native allocations and hard audio-callback behavior are unvalidated. TCN/non-GRU ONNX uses fixed batch=1 and fixed exported block size. Carry `next_history` into the next call and zero-pad/crop a final short block in a host adapter. GRU export is deliberately rejected until an explicit persistent-hidden-state adapter exists.

**Limitations:** tiny synthetic sample; one seed; mismatched parameter counts; no aliasing analysis; no trained Temporal FiLM/reference encoder; no perceptual results; no native C++ or VST3. GRU variants failed initial Python CPU screening. Float-WAV rendering preserves the model's output without clipping or limiting and reports peak level.

**Distribution:** no external datasets, weights, SDKs or copied research implementations are bundled. Repository owner must choose a code/weights license and verify dependencies before a release.
