# Ordered connection training and native acceptance

The fixed local run completed on 2026-10-16. All native state and integration gates passed. The model now has outputs for 64 numbered connections and optional mapping curves, trained from original source audio. **This is a capability integration, not a qualified model:** the small training run still creates many spurious connections and performs worse than training retrieval on the five-case sound panel. Strict tiny-overfit remains false; no model is promoted.

## Preserved state and native evidence

Each connection retains its literal slot, source, destination, mapping presence and five indexed scalar controls. References to another slot's amount remain unchanged. The model represents all 32 source names and 431 destinations in the pinned full contract, with 6,896 continuous values and 802 categorical heads. The qualified banks give 5,668,542 trainable parameters. Older codecs retain their saved contracts.

| Native check | Result |
|---|---|
| Source vocabulary | 32/32 states accepted |
| Destination vocabulary | 431/431 states accepted |
| Numbered positions with mapping and indexed controls | 64/64 accepted, each repeated exactly |
| Self-amount reference, LFO feedback and amount cycle | All three accepted, each repeated exactly |
| Dense 64-slot state | Accepted and repeated exactly |
| Mapping on an empty connection | Both attempts rejected by exact state verification, as expected |
| Total | 601 attempts: 599 accepted, two expected rejections; 68 accepted repeat pairs exact |
| Resources | 987.77 seconds; 543.85 MiB peak process memory; no sampling errors; below 8 GiB |

The initial eight-attempt probe exposed Vital 1.6.4 dropping a line mapping attached to an empty connection: six accepted renders and two state-readback errors. That failed probe remains retained. The encoder now explicitly refuses this source state, and the decoder emits mappings only for connected slots. Latent unused predictions remain in saved targets. The final matrix records the two matching errors as expected rejections; unrelated errors still fail. Full scalar and nested-state readback is unchanged.

All correctness repeats fix oscillator and sampler random phases while preserving actual curve phases. Vocabulary fixtures certify state acceptance and valid audio, not audible effectiveness for every source/destination or correctness of every possible feedback graph. Several source types require performances absent from these fixtures; inactive destinations are not thereby musically qualified. No final-test state was rendered.

## Original-audio training

The population preserves 256 original root assignments: 205 training, 27 validation and 24 reserved test. Its five authored families are direct, parallel, indexed amount, LFO feedback and sixteen-connection dense patterns. Master volume is stored as `3844` throughout the declared curriculum to leave headroom. The actual audio is not rescaled after rendering.

Training renders original authored LFO and mapping shapes. Separate tensors hold the original 97-point samples as labels; only audio-derived timbre and frozen Basic Pitch features enter the model. All 64 retained pairs of LFO/map label tensors were independently checked against their original source recipes. Recovery verifies these labels and reproduces model, optimizer and random-number state.

| Measurement | Result |
|---|---|
| Updates | 400 CPU updates, batch 16, four threads; 6,400 optimizer draws |
| Collection | 64/64 training renders accepted; no clipping rejections or errors |
| Distinct retained source states/audio buffers | 64/64 |
| Replay | 2,474,240 feature bytes + 198,656 LFO-label bytes + 1,589,248 map-label bytes = 4,262,144 bytes, within 8 MiB |
| Training/collection time | 387.12 seconds, including 15.82 seconds feature extraction and 220.93 seconds optimizer work |
| Peak process memory | 1,145.96 MiB; no sampling errors; below 8 GiB |
| Saved-target correctness | 15/15 native renders; all five families; exact repeated/restored audio |
| Export/reload | Model weights and codec identity match; frozen frontend unchanged |

Retained family counts were direct 16, parallel 15, indexed amount 11, feedback 12 and dense 10. Connected examples reached 63/64 positions; slot 46 was absent from this bounded cache. All 64 positions are implemented and independently native-checked, but this run does not claim training coverage of every position. Each numbered LFO was active in 20–24 retained examples.

| Complete retained-set metric | Initial | Final |
|---|---:|---:|
| Combined loss | 1.549441 | 0.325740 |
| Continuous MAE | 0.051067 | 0.033217 |
| Weighted categorical accuracy | 77.6366% | 88.1137% |
| Original LFO reconstruction MSE | 0.113018 | 0.046454 |
| Original mapping reconstruction MSE | 0.012213 | 0.002416 |

Presence classes receive equal total loss weight so empty slots cannot dominate by count. This does not itself guarantee good predictions. On the retained training set, the model detected 205 of 294 connected slots, missed 89, and emitted 1,653 false connections: **69.73% recall and 11.03% precision**. It matched the complete present/source/destination triple on 95/294 target connections. Source labels were correct on 165/294 connected targets and destinations on 233/294, measured independently of presence. Aggregate accuracy is therefore insufficient evidence of learned routing. No weight sweep or extra seed study was used to conceal this gap.

## Native sound panel

One frozen validation state per family was rendered under reference and reserved-transfer performances. All 60 planned renders were attempted and accepted, including ten model predictions. Literal source repeats were exact. The label reconstruction is an evaluator-only representation diagnostic after model inference; it never enters the model input.

| Mean distance from original sound; lower is better | Reference, 5/5 scored | Transfer, 5/5 scored |
|---|---:|---:|
| Label reconstruction: representation error | 0.001398 | 0.002329 |
| Audio-only model | 1.727061 | 2.055009 |
| Training-only retrieval | 1.174706 | 1.387540 |
| Fixed template | 3.204470 | 3.975992 |
| Literal source repeat | 0 | 0 |

Prediction error is far larger than representation error on these small, conservatively modulated examples. Retrieval is better on both performances. The passing integration gate proves valid state, audio, training and export; it is separate from strict overfit, broader graph learning, quality and release gates. These results do not justify dropping modulation from the patch model.

The [native plan](data/routing_native_plan.json), [native summary](data/routing_native_summary.json), [training plan](data/routing_training_plan.json), [training summary](data/routing_training_summary.json) and [saved-target checks](data/routing_training_codec_summary.json) preserve identities, failure details, runtime source hashes, LF-normalized source hashes and artifact hashes. Runtime sources were unchanged throughout each final run. No waveform archive, trained weights or third-party asset payload is committed. Checks overlapped on the local machine, so timings are not isolated throughput benchmarks.

The affected regression command passed 140 checks in 483.90 seconds. Two subsequent rejection-classification checks passed in 6.69 seconds, proving unrelated errors cannot count as expected rejection. Ruff and whitespace checks pass. These are separate test commands, not a single 142-case run.

Free local CPU remains the qualified path; Linux/Colab, broader combined training and final release quality remain open. GitHub Actions stayed disabled. Custom warps and random seeds are the next independent capability in the [completion ledger](../IMPLEMENTATION_STATUS.md). Reproduction and approximation limits are in [ORDERED_ROUTING.md](../ORDERED_ROUTING.md).
