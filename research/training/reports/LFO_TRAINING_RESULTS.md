# Eight modulation curves learned from original audio

Completed locally on 2026-10-16. The model now has eight Era2 curve outputs, original-audio training, original-curve reconstruction supervision, activity masks and a reloadable inference bundle. Training and export completed. The native complete-audio gate **failed on one clipped source performance**, and strict tiny-overfit remains false. No model is promoted.

## Completed implementation checks

The model receives timbre and frozen Basic Pitch features only. Original curve samples are separate training labels. All 64 retained label tensors exactly match their original source recipes; none is substituted with a code-reconstructed curve. Checkpoints account for these tensors, reject altered labels and reproduce model, optimizer and random-number state after interruption. A collected-only checkpoint can seed a fresh learner under the same pool/codec contract.

The fixed authored population preserves all 256 original scalar roots: 205 train, 27 validation and 24 reserved test. Each LFO is audible in the retained training data, with 5–11 examples per numbered slot. Sixteen declared connections feed oscillator/sampler level; their structure is fixed in this increment. Learning arbitrary ordered connections remains next.

| Measurement | Final result |
|---|---|
| Model | 1,012,158 trainable parameters; frozen frontend unchanged |
| Outputs | 688 continuous values, 546 categorical heads; eight distinct LFO codes |
| Training | 1,000 CPU updates, batch 16, four threads; 64 distinct source states/audio buffers |
| Training native attempts | 67 attempted, 64 accepted, three clipping rejections, zero errors |
| Retained feature tensors | 2,474,240 bytes |
| Original curve label tensors | 198,656 bytes; 2,672,896 bytes combined replay |
| Training/collection time | 574.52 seconds, including 19.28 seconds of feature extraction |
| Peak process memory | 1,005.20 MiB, below 8 GiB; no sampling errors |
| Saved-target/native checks | 24/24 renders accepted; all eight families; exact repeated/restored audio with oscillator and sampler random phases fixed |
| Export/reload | Restored model state and codec identity match |

This is a bounded overfit/integration run: 16,000 optimizer draws reuse 64 source clips. It is not the broad data-scaling run. All 775 scalar fields retain outputs, but the curriculum varies a selected subset and one active LFO per patch. It does not establish recovery of every control or simultaneous complex modulation.

| Complete retained-set metric | Initial | Final |
|---|---:|---:|
| Combined loss | 0.749380 | 0.028229 |
| Continuous MAE | 0.047438 | 0.024432 |
| Categorical accuracy | 91.9976% | 99.7426% |
| Soft curve reconstruction MSE against original samples | 0.118325 | 0.011630 |

Tiny-overfit requires 100% categorical accuracy, so it remains false. Loss-reduction and continuous-MAE thresholds pass. Actual active LFO index accuracy is 1,077/1,088; the aggregate also includes many fixed scalar labels. Hard predicted curves have mean training RMSE `0.129829` against the original samples, versus `0.033700` for the target codes. Indices alone do not establish accurate curves: phase/gain and hard-choice reconstruction still leave a substantial prediction gap.

## Native sound validation

The panel freezes one validation state per LFO slot, each with reference and reserved-transfer performance. It planned 96 renders; 91 were attempted and 90 accepted. The eighth source's transfer render clipped at peak `1.27299`, so five dependent renders were skipped. All 15 scorable model predictions rendered successfully, with no state-readback failures. The three rejected training clips had peaks `1.35299`, `1.52622` and `1.09818`. All attempts remain charged, and clipping thresholds are unchanged.

| Distance from original source | Reference mean (scored/planned) | Transfer mean (scored/planned) |
|---|---:|---:|
| Label-code reconstruction: representation error | 0.03020 (8/8) | 0.01943 (7/8) |
| Audio-only model prediction | 2.29360 (8/8) | 2.90646 (7/8) |
| Training-only retrieval | 1.43183 (8/8) | 3.97943 (7/8) |
| Fixed template | 5.54548 (8/8) | 50.13387 (7/8) |
| Literal source repeat | 0 (8/8) | 0 (7/8) |

Model-to-code-reconstruction distance is 2.28264 on reference and 2.89935 on transfer. In this small panel, prediction error is much larger than representation error. The model is worse than retrieval on reference and better on transfer; one source performance is unscorable. These results do not pass a quality or generalization gate. Source repeats render the literal original patch, avoiding another lossy encoding. Label-code reconstruction is an evaluator-only diagnostic and never an inference input.

## Periodic decoder repair and retained failed runs

The first wiring rendered code-reconstructed curves as inputs. That prototype was stopped after 24 saved-target checks and 16 training renders, before any optimizer update. The source/target contract now explicitly renders original authored curves, binds each code label to that source and supervises original sampled values. No prototype features were reused.

The first original-source run collected 64/64 accepted clips but failed before initial metrics/optimization. Float32 periodic arithmetic could round a remainder to `97`, giving an out-of-bounds tensor index. The edge test also exposed a shared NumPy error: subtracting an already-wrapped index could produce an invalid interpolation fraction at extremely small phases. Tensor interpolation now separates integer displacement from its fraction; NumPy computes the fraction before wrapping the index. Regression checks cover all 97 lattice boundaries, adjacent floats and tiny phases, enforce bounded interpolation, and retain finite gradients. Hard tensor decoding matches the saved-code decoder within the predeclared `2e-5` numerical bound. Native saved-target audio checks remain exact.

Changing the shared decoder invalidates older bank fingerprints. The bank was rebuilt from training roots and requalified in **80/80** native renders, with all 32 original/code repeat pairs exact. It took 164.13 seconds and peaked at 856.45 MiB. The new bank ID is `2aac62a02d780ee330173c1e1f3cbf5b282c9cf5c7ee17d0d638148f72cc0c31`. Old bank/cache contracts are refused; the final run collected fresh source features. The [bank plan](data/lfo_boundary_bank_plan.json) and [bank summary](data/lfo_boundary_bank_summary.json) retain this evidence.

The [training plan](data/lfo_training_plan.json), [summary](data/lfo_training_summary.json) and [saved-target checks](data/lfo_training_codec_summary.json) retain source hashes, root identities, all failures, coverage and artifact hashes. Raw hashes identify the local runtime; additional LF-normalized hashes link its source text to Git across Windows line endings. The training package and runner were unchanged throughout the final run. The bank loader separately verifies its normalized decoder source hashes.

A 274-check complete training-suite command passed before the final interpolation/collection repairs. All 79 final affected checks passed after those repairs, including the LFO bank, training, source-label recovery and Era2 component/alignment/dataset paths. The focused periodic/hard-decoder/recovery command also passed all three checks. Ruff and whitespace checks pass. These are separate commands, not one clean 275-case run. Local checks overlapped part of collection, so elapsed time is not an isolated throughput benchmark.

No final-test audio, waveform archive, third-party asset payloads, paid compute, Actions execution or automatic merge was used. Source headroom, complete native validation, strict overfit, broad curriculum and model quality remain open. Learned ordered routing, custom warps/seeds, wider performance coverage and data scaling remain on the full plan. Reproduction and contracts are in [LFO_TRAINING.md](../LFO_TRAINING.md).
