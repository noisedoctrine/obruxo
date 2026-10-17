# Combined control, asset and performance pipeline

Completed locally on 2026-10-17. The combined authored curriculum varies 769 of the 775 scalar controls, uses every registered asset choice and activates all synthesis families together. Native state/effect checks and the bounded training/recovery integration pass. These results establish source coverage and pipeline behavior. Useful reconstruction, strict tiny-overfit and release quality remain open; no model is promoted.

## Coverage and native behavior

The frozen roots remain 205 training, 27 validation and 24 reserved test. The new child states preserve their original ancestry and split assignments. Curves, mapping shapes and seeds retain their original source bindings. A new codec identity records every raw range/choice, the four-second tail, sparse/dense connection policy and native audio-destination domain.

| Declared training-recipe coverage | Previous expression pool | Combined pool |
|---|---:|---:|
| Scalars varied under supervision | 214 | 769 |
| Scalars constant under supervision | 374 | 6 |
| Scalars always masked | 187 | 0 |
| Wavetable choices per oscillator | 1 / 1 / 1 | 20 / 20 / 20 |
| Enabled sampler choices | 0 | 11 |
| Named modulation sources observed active | 14 | 32 |
| Connection positions observed active | 64 | 64 |
| Effect families enabled | 2 | 9 |

The six fixed fields are bypass, host-clock tempo storage and four display controls. These counts describe supervision eligibility. They do not prove audibility or learning for every parameter combination. The [complete inventory](data/combined_curriculum_coverage.json) observes 326 of the 431 modulation destinations under active supervision; broader graph/range coverage remains open. Sparse scenes keep the original connections and add two sources; dense scenes include all named sources and fill all 64 positions.

| Native qualification | Result |
|---|---|
| Combined sources | 29 training fixtures, including every registered choice in each active slot, each repeated |
| Family effects | Fifteen before/after/repeated fixtures: all nine effects, sampler content, voice/polyphony, portamento, LFO/random playback and a secondary envelope |
| Total | 103/103 accepted; all seven gates pass |
| Original labels | Independent raw-source encoding matches every selected target |
| Repeatability | Exact float32 audio with four generator random phases fixed and a fresh process per request |
| Resources | 238.10 seconds inside qualification, excluding initial pool validation; 600.14 MiB peak process memory; no sampling errors |

The separate [destination qualification](data/destination_native_summary.json) makes 100 attempts across all three oscillator slots and the sampler. It accepts twenty standard routes and correctly rejects eighty exact-zero outputs for inactive destination values 5–14, with all 775 requested scalars verified. This confirms a measured DSP limitation, rather than rejecting an implemented component because of a model score. All nine effects remain available through the normal effect chain; the new codec predicts the five functioning audio destinations without remapping unsupported inputs. Details and the explanatory source link are in [COMBINED_CURRICULUM.md](../COMBINED_CURRICULUM.md).

## Failed checks and corrections

The initial four-scene probe accepted only two of eight renders because the advertised destination range included inactive buses. The first complete combined panel then accepted 101/103 attempts. One source was below the unchanged silence threshold in both repeats, and random modulators exposed process-scoped state when workers were reused. Original labels and all family-effect checks passed. The [initial plan](data/combined_initial_native_plan.json) and [summary](data/combined_initial_native_summary.json) retain those failures; their source hashes resolve to commit `561cf3e`.

The final source policy raises stored master gain from -35..-30 dB to -29..-24 dB, preserving explicit volume labels and render levels. Sparse scenes retain sparse connections. Correctness rendering resets process-scoped random state through fresh processes while preserving all requested random routes and fixing the oscillator/sampler random phases. The fifteen independent family fixtures have exactly the same 45 request IDs in both panels. Random playback now repeats exactly. The final [plan](data/combined_native_plan.json), [summary](data/combined_native_summary.json) and [evidence manifest](data/combined_evidence_manifest.json) verify the corrections without relaxing audio QA or repeat tolerances.

## On-demand training and recovery

The integration renders audio inside the existing training loop and retains features and original labels. It uses four CPU threads, a replay limit of 8 MiB, at most 128 acknowledged native attempts and twelve updates. A saved checkpoint at update six is resumed and compared with the uninterrupted twelve-update run. Recovery uses the retained features and makes no new renders.

| Measurement | Result |
|---|---|
| Source collection | 67 acknowledged attempts: 64 accepted, three rejected, zero errors |
| Rejections | Sequences 6 and 15: exact silence; sequence 45: RMS `5.960843e-7`, below the `1e-6` cutoff; all retained in the denominator |
| Distinct training states/audio buffers | 64 / 64 |
| Reference-performance exposure | All eight; counts range from three to thirteen |
| Actual retained scalar exposure | All 769 variable controls vary under their supervision masks; six fixed controls |
| Actual retained asset exposure | 19 / 19 / 16 wavetable choices by oscillator; all eleven sample choices; every effect family enabled |
| Model | 5,725,689 trainable parameters; 7,187 continuous outputs and 898 categorical heads |
| Model inputs | Audio-derived timbre and frozen Basic Pitch only |
| Retained replay | 4,336,640 bytes: 2,474,240 features + 198,656 LFO + 1,589,248 mapping + 74,496 custom-curve labels |
| Training segment | 221.56 seconds: feature extraction 30.36 seconds, optimizer work 6.48 seconds, checkpoint work 145.80 seconds |
| Peak process-tree memory | 1,226.45 MiB, below 8 GiB; no sampling errors |
| Exact recovery | Model, optimizer, Python/NumPy/Torch and batch RNG, replay and cursor match |
| Export/reload | Model state and codec match; frozen frontend unchanged |
| Gates | All twelve pipeline gates pass; strict tiny-overfit remains false |

The [training plan](data/combined_training_plan.json), [summary](data/combined_training_summary.json) and [retained-exposure analysis](data/combined_training_exposure.json) separate prospective recipes from consumed examples. The twelve updates prove functioning learning and recovery. They do not establish useful reconstruction or generalization. No native prediction-quality comparison is claimed here.

Checkpoint writes account for roughly two thirds of this training segment. That is a concrete free-compute bottleneck before scaling; native rendering remains offline. Warmup persistence should avoid repeatedly serializing unchanged model state while preserving durable feature/cursor recovery. The three source-QA failures also need causal inspection before broad source robustness can pass. These are pipeline/data tasks, not another small parameter-weight experiment.

The full local training suite passed 332 tests at the saved initial implementation. After the final gain/connection changes, all 33 affected checks passed, including legacy expression contracts, original-label preservation, sparse/dense sources, legal output domains, pool/CLI round trips and exact training recovery. Ruff, whitespace and staged-source fingerprint checks pass. Native checks, regression tests and static auditing overlapped, so durations are not isolated throughput measurements. The environment is Windows 11, Python 3.12.13, DawDreamer 0.8.3 and Torch 2.12.1+xpu, executing on CPU.

Only metadata is committed. Factory assets, waveforms and weights remain local. Final-test audio and GitHub Actions are untouched. Free local CPU remains qualified; Linux/Colab, input-layout gaps including one-point curves, complete graph/range coverage, scaling beyond the frozen roots, useful validation quality and release gates remain open. Reproduction commands are in [COMBINED_CURRICULUM.md](../COMBINED_CURRICULUM.md).
