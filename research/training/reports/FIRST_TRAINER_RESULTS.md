# First trainer results

Runs and tests: 2026-10-05. Report completed and exported-model checksum reverified: 2026-10-08.

**The first audio-to-patch learning loop works on the local CPU.** A 355,806-parameter model overfits 32 native-rendered examples, exports valid pilot presets and resumes exactly from retained features after a deliberate interruption. Its four-state validation results do not yet establish useful reconstruction quality: training-only retrieval performs better on reference audio. No paid compute, GPU training or Colab session was used.

Reproduction instructions are in [TRAINER.md](../TRAINER.md). The [machine-readable report](data/first_trainer_summary.json) contains configurations, identities, timings, metrics and artifact hashes.

## Native loading defect found and corrected

The first training attempt produced only four distinct audio buffers from 32 views of 30 patch states: one buffer per MIDI performance. Native state inspection showed that Vital ignored the requested patch and retained its default controls. The authored template had inherited Vita's synthetic `synth_version: 99999.9.9`; the pinned native Vital 1.6.4 does not accept that marker.

Newly authored pilot presets use `1.0.8`. The general renderer explicitly bridges only Vita's synthetic marker to that reviewed version, records the transformation in diagnostics and native QA, and rejects arbitrary future versions. It saves the native state back after loading and verifies the requested scalar controls, LFOs and modulation routes before rendering. The native regression now requires a requested level reduction to produce a corresponding RMS reduction. This readback does not certify arbitrary sample/wavetable conversions.

The renderer identity now ends in `state-verified-v2`; the codec is `authored_osc_filter_env_v2`. The regenerated 256-state pool has **205 training, 21 validation and 30 test states**. Its 32 training views produced 32 distinct audio buffers. Existing corpus files and reviewed schema artifacts were not edited. Historical pools/checkpoints/cursors fail current compatibility checks and must not be silently reused.

The [earlier pilot report](ONLINE_PILOT_RESULTS.md) and its JSON are marked superseded for native patch fidelity. Their QA/timing observations described the incorrect render path. The corrected codec check accepted **18/18 renders**, covered all six categorical combinations, and passed all six reconstruction and source-repeat comparisons under the existing tolerances.

## Tiny overfit and compute

The model sees stereo full-band spectral features and frozen Basic Pitch dense features from the same audio. It receives no MIDI, source identity or oracle performance labels. Its scope remains one reviewed init oscillator, optional filter 1, envelope 1 and the declared level/pan/unison controls. All other state is fixed by the codec.

| Measure | Result |
|---|---:|
| Native training views / distinct training states | 32 / 30 |
| Trainable parameters | 355,806 |
| Optimizer updates / batch size | 800 / 16 |
| Initial / final total supervised loss | 1.871843 / 0.001185 |
| Final active-control normalized MAE | 0.024533 |
| Final categorical accuracy | 100% |
| Predictions passing schema validation | 32 / 32 |
| Frozen Basic Pitch weights and buffers | Unchanged |
| Feature replay payload | 1,237,120 bytes |
| Initial render/400-update segment | 74.80 s |
| Resumed 400-update segment | 3.27 s |
| Total optimizer time across 800 updates | 5.66 s |
| Total feature extraction time | 2.96 s |
| Peak sampled process-tree RSS | 898,834,432 bytes, about 857 MiB |
| Training WAV files written | 0 |

Training used float32 PyTorch 2.12.1+xpu on **CPU**, with four Torch CPU threads, AdamW at 0.003, no scheduler/mixed precision, and finite-gradient/weight checks. The tiny gate requires normalized MAE below 0.05, perfect category accuracy and final loss below 5% of initial loss; it passed. This is supervised parameter bootstrap, not perceptual-loss training.

A separate online smoke interleaved **11 accepted fresh native renders with eight optimizer updates**, using an eight-example FIFO feature buffer. It took 25.36 seconds, about **0.43 accepted clips/second**, and peaked at 939,732,992 sampled process-tree RSS bytes, about 896 MiB. It had no QA rejections or render errors. Its short training loss is not learning-curve evidence; it verifies the coupled online path. Initial and final online replay metrics describe different FIFO contents and should not be compared as a fixed-set learning curve.

RSS is the sampled sum of parent/child resident sets, not unique physical memory, and can count shared pages twice. Native rendering dominates the elapsed time. This narrow stage fits the local machine comfortably; moving only the neural network to a faster GPU would leave that bottleneck. The Linux native renderer and Colab interruption path remain unqualified.

## Interrupted recovery

Using the actual retained features from native audio, execution deliberately raised an interruption after optimizer step 605. The last persisted checkpoint was step 600. Restoring it and completing step 800 matched the uninterrupted continuation exactly in model tensors, optimizer state, global RNG state, batch RNG, acknowledged render cursor and replay contents. No fresh features were extracted during recovery.

The test suite separately interrupts the online path after render acknowledgement but before its optimizer update. Recovery neither skips that view nor ingests another replacement view. Changed contracts and tampered replay membership are rejected. These are same-runtime CPU guarantees for retained features; newly generated Vital audio and cross-device execution are not claimed bit-identical.

## Held-out real-render smoke

Four unique validation states were selected deterministically before scoring. Each predicted patch came from one reference audio clip and was reused for one reserved transfer performance. True MIDI was supplied only to the evaluator's renderer. The final test split was not evaluated.

Distances below are mean stereo multi-resolution STFT error, combining spectral convergence and log-magnitude MAE at FFT sizes 512, 2,048 and 8,192. Lower is better. This diagnostic is not a calibrated perceptual score.

| Method | Reference distance | Transfer distance |
|---|---:|---:|
| Model, one prediction | 1.4982 | 1.5184 |
| Fixed pilot template | 1.5398 | 2.0127 |
| Nearest training replay example | **1.3204** | 1.5460 |
| Source repeat, diagnostic floor | 0.0000 | 0.0000 |

All **40/40 evaluation renders passed QA**, including source, model, baseline and repeat renders. Every method/role scored all four planned cases. Retrieval used only the 32 retained training examples, covering 30 training states, with fixed feature-distance scaling. It did not use held-out targets to rank candidates. Source repeats happened to be identical in these cases; that is not a general determinism guarantee.

The model is slightly better than the fixed template on reference audio and better on transfer in this small sample. Retrieval beats it on reference audio; the transfer difference versus retrieval is too small and the sample too limited to support a quality claim. The checkpoint is therefore a **pipeline proof**, not a selected release model.

## Artifacts and verification

Local ignored artifacts:

- [Inference bundle](../outputs/tiny-train-v2-complete-20261005/model.pt): 1,585,110 bytes; includes trained and frozen weights plus executable-contract descriptions.
- [Training checkpoint](../outputs/tiny-train-v2-complete-20261005/checkpoint.pt): 5,732,669 bytes; includes optimizer/RNG/replay state.
- [Pilot pool](../outputs/authored-pilot-v2-20261005/pool.json) and [validation summary](../outputs/tiny-validation-v2-20261005/summary.json).
- [Example predicted patch](../outputs/tiny-validation-v2-20261005/prediction-000.vital), loaded successfully by the native evaluator.

Inference-bundle SHA-256: `80fa2b2093810df9f5fdf2ec5fedc6ede10162d5c81652eead50571b2acdae8c`.

Validation completed: **124 non-native tests passed**, **three native integration tests passed**, and Ruff passed across the changed training/renderer code and tests. Checks include frozen statistics, stereo information, masked gradients, exact interrupted recovery, online consumption ordering, inference from integer PCM, overwrite refusal, render-budget exhaustion and failed-evaluation denominators. CPU CI was added, but no remote CI run is claimed. The native integration run emitted a pytest cache-permission warning; its tests passed.

The next increment is now measured in the [full-pilot report](FULL_PILOT_RESULTS.md): all 205 training states, stronger retrieval, 21 validation states and fixed checkpoint comparisons. The original comprehensive training plan still governs the eventual broad model.
