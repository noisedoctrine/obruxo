# Train-only LFO assets and native reconstruction

Completed locally on 2026-10-16. The existing Era2 W8D16 phase/gain representation now has a training-split-safe fitting path, saved codebooks, validated codes and a decoder that needs no original curve or topology label. All seven correctness/resource gates pass in the final run. This does not yet add audio-to-code model training or pass a perceptual quality gate.

| Measurement | Result |
|---|---|
| Source population | 256 authored variants retain the original 205 train / 27 validation / 24 test roots |
| Fitting input | 205 training presets, 1,640 unique LFO curves across eight numbered slots |
| Bounded fit | 64 unique training curves, selected deterministically by hash |
| Representation | 32 base atoms; 16 residual layers, eight atoms each; phase and residual gain; 97 inclusive points; final clipping only |
| Numeric dictionary storage | 62,080 bytes as float32, before provenance/serialization |
| Output accounting | 193 outputs per LFO; 1,544 across eight slots, before other controls |
| Saved-bank identity | `21c49269186deec1142705404e291379e64b86b43154ceb11e8072ec297820dd` |
| Native panel | Eight cases across all eight slots: four train and four validation; held-mid and held-low performances |
| Native attempts | 80/80 accepted; all 775 scalar controls and full nested state verified |
| Repeat correctness | All 32 original/code repeat pairs are exactly equal with oscillator/sampler random phases fixed |
| Decoder correctness | Reloaded-code reconstruction exactly matches the existing encoder reconstruction on all eight cases |
| Elapsed / peak memory | 134.15 seconds / 853.23 MiB process memory; no sampling errors; below 8 GiB |

The reconstruction measurements separate the fixed-grid approximation from the dictionary approximation. Curve RMSE is in stored `[0,1]` units. Sound distance uses the existing multi-resolution spectral metric; lower is closer, without a calibrated perceptual pass threshold here.

| Mean diagnostic | Four training cases | Four validation cases |
|---|---:|---:|
| Original-to-97-point dense curve RMSE | 0.000508 | 0.000121 |
| Sampled curve-to-code RMSE | 0.042919 | 0.032841 |
| Original-to-grid native sound distance | 0.010095 | 0.001660 |
| Grid-to-code native sound distance | 0.129133 | 0.120041 |
| Original-to-code native sound distance | 0.132020 | 0.120177 |

Each sound mean covers both performances. Maximum sampled curve-to-code RMSE is 0.054004 on the training panel and 0.041375 on validation. The cases use only two source preset roots, alternating slots between them. They establish a bounded eight-slot native diagnostic, not representative generalization, all possible routing behavior or superiority of one split. The original and reconstructed sounds differ measurably. Larger dictionaries, representation tuning and prediction from audio were not tested in this increment.

The first native prototype accepted 48/80 attempts. All 32 original/repeat requests failed strict LFO readback because the authored source coordinates and powers were not serialized at native float32 precision. Grid/code requests already loaded. The repair casts authored points/powers before state hashing and dictionary fitting. It preserves the complete original root manifest and every split assignment. The final run uses new child hashes under those same roots. Native thresholds and readback checks were not relaxed. The failed prototype remains recorded in the [summary](data/lfo_bank_summary.json).

The [plan](data/lfo_bank_plan.json) pins the source implementation, member identities, panel, performance bank, phases, 64-curve budget and 80-attempt budget. The [summary](data/lfo_bank_summary.json) records fitting provenance, code targets, per-case measurements, native request/audio hashes, all gates and the failed prototype. Source hashes match the completed worktree. The artifact loader rejects changed codebooks/policies, altered decoder sources, invalid dimensions or values, missing zero atoms and any held-out fitting membership. Supplying a held-out or unknown preset to fitting fails before dictionary construction.

The affected training/Era2 command passed 37 checks. A separate regression check passes for native fixture precision and exact preservation of all 256 roots. Ruff and whitespace checks pass. The new CLI also proves saved-bank reload and refusal of held-out input. These checks do not require a GPU. The native run used the pinned Windows Vital/DawDreamer profile on the local CPU; Linux/Colab remains unqualified.

No final-test audio, waveform archive, model promotion, paid compute or Actions execution was used. The codebooks were fitted only on the declared authored training curves, and no external LFO assets or source preset payloads are committed. Input lines currently require at least two control points, and a full repository runtime is still required. Connecting the curve outputs, reconstruction loss and activity masks to the inverse model is next, followed by custom warps/seeds and ordered routing. See [LFO_BANK.md](../LFO_BANK.md) and the [completion ledger](../IMPLEMENTATION_STATUS.md).
