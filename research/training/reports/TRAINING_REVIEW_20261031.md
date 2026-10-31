# Training review: recover the smallest useful learning loop

Reviewed 31 October 2026 against main `ced6b0b`, the plan's Git history, recorded experiment results and the 100 merged PRs numbered 39–139 (there is no PR 62 in that returned history). This is a planning review, not a new experiment or a model promotion. The [evidence snapshot](data/training_plan_review_20261031.json) records PR identities, recomputed timing totals and the running fit's dated status.

## The original aim

The [first committed training plan](https://github.com/noisedoctrine/obruxo/blob/d9f2790/research/modelling/TRAINING_PLAN.md), prepared from 4 October and committed on 14 October, recommended a complete small model, improvement against real Vital renders, then controlled expansion. The earlier [minimal training design](../../modelling/simple_training/README.md) was even narrower: one oscillator, octave and level, freshly rendered audio and a tiny predictor. Its two-step smoke proved connectivity, not useful learning.

The product aim has remained: **one isolated recording in; one playable patch out that sounds similar and behaves sensibly when played differently.** The [research tracker](../../modelling/RESEARCH_TRACKER.md) makes sound similarity primary, source-parameter supervision auxiliary, and inference audio-only. Local, open-source operation and on-demand training audio are part of that aim. The intended release is a useful instrument-reconstruction tool, not a complete inverse of every Vital configuration.

The original plan also contained a sequencing problem. It called perceptual quality primary, but placed synthesis expansion in Stage C and perceptual credit assignment in Stage D. Its future codec discussion specified all 64 route slots and eight complex LFOs. That made coverage look like a prerequisite for useful learning. We should acknowledge this weakness in the plan itself, rather than claim implementation simply ignored an otherwise unambiguous roadmap.

## How the work evolved

| Period / PRs | Work delivered | What the evidence means |
|---|---|---|
| Contracts [#37](https://github.com/noisedoctrine/obruxo/pull/37); pilot/research [#39](https://github.com/noisedoctrine/obruxo/pull/39)–[#42](https://github.com/noisedoctrine/obruxo/pull/42) | Identity/splits, on-demand rendering, compact model, recovery, real-render baselines, 205/404-state learning and replication | The complete narrow path works and has promising average gains. Reused small validation panels and mixed replication do not establish reliable usefulness. |
| Independent directions [#43](https://github.com/noisedoctrine/obruxo/pull/43)–[#50](https://github.com/noisedoctrine/obruxo/pull/50) | Broader alpha, paired identity learning, sound proxy, render-selected targets, ranking, state-conversion speed, worker reuse and conditioning comparison | Useful bounded investigations. Contrastive learning and attention did not demonstrate reconstruction gains. The proxy failed real-audio guidance. Their implementation was not a reason to advance model quality. |
| Coverage expansion [#51](https://github.com/noisedoctrine/obruxo/pull/51)–[#71](https://github.com/noisedoctrine/obruxo/pull/71) | Full scalar/asset support, eight LFOs, 64 routes/maps, custom curves/seeds, broader sources and expression | Substantial reusable infrastructure, but prediction scope grew ahead of reconstruction proof. Codec fidelity and a large output layer became confused with a learning milestone. |
| Scope/objective correction [#72](https://github.com/noisedoctrine/obruxo/pull/72)–[#85](https://github.com/noisedoctrine/obruxo/pull/85) | Actual 14-value basic head, spectral objective, helper qualification, native gradients and startup settling repair | A necessary return to the small task. Learned helpers failed; measured native gradients passed bounded checks. The corrected renderer invalidated reuse of incompatible old targets. The first sound-trained candidate still missed its quality target. |
| Input and training corrections [#87](https://github.com/noisedoctrine/obruxo/pull/87)–[#124](https://github.com/noisedoctrine/obruxo/pull/124) | Category derivatives, wider inputs, reciprocal losses, correction branches, rehearsal, measured acceptance and trainable input features | Several real defects were fixed and some results improved. Nevertheless, repeated costed fits did not establish robust shared reconstruction. We increasingly optimized details of a difficult training setup before resolving data coverage, affordable learning exposure and the basic shared-learning question. |
| Inference, listening and component diagnosis [#125](https://github.com/noisedoctrine/obruxo/pull/125)–[#134](https://github.com/noisedoctrine/obruxo/pull/134) | Audio-only export/runtime, notices, confirmation reservation, listening fixes, level analysis and direct fitting | Reusable release foundations and important diagnostic evidence. They do not complete training. Loudness-confounded human comparisons could not establish a tone winner. Direct fitting finally demonstrated that three difficult sounds can be fitted within the existing controls. |
| Separation and data coverage [#135](https://github.com/noisedoctrine/obruxo/pull/135)–[#139](https://github.com/noisedoctrine/obruxo/pull/139) | Controlled probes, calibrated removal losses, shared three-sound comparison, balanced note/duration bank and running fit | The branches are not demonstrably separated; tested auxiliary settings barely help. Balanced data addresses a concrete confound, but its fit result is pending. All five PRs are merged; merge status is not experiment completion. |

PR 86 prepared blind listening, and PRs 99, 101, 106 and 107 repaired export/execution/recovery. Those are correctness and usability work, not additional quality wins. Dates and the complete PR titles are in the snapshot. The [18 October coverage snapshot](../../modelling/TRAINING_PLAN_COVERAGE_20261018.md) and [pre-review plan](https://github.com/noisedoctrine/obruxo/blob/ced6b0b/research/modelling/TRAINING_PLAN.md) preserve the successive priorities.

## What we have actually established

| Question | Current answer | Evidence |
|---|---|---|
| Can legal small patches be rendered, exported and recovered? | Yes, within the qualified local runtime and fixed-phase contract. | [Renderer settling](NATIVE_PREROLL_RESULTS.md), [production choices](PRODUCTION_NATIVE_RESULTS.md), [clean runtime](CLEAN_BASIC_RUNTIME_RESULTS.md) |
| Can the present sound objective guide individual controls? | Yes on bounded checks; this does not guarantee successful network training. | [Settled gradients](SETTLED_NATIVE_GRADIENT_RESULTS.md) |
| Can the existing vocabulary fit the three difficult sounds? | Yes. Individual native optimization reduces joint error by 86%, 59% and 41%, beating their settings-trained controls. It uses evaluator performances and does not train a shared predictor. | [Direct control fitting](DIRECT_BASIC_CONTROL_RESULTS.md) |
| Can one network infer useful patches across different recordings? | Partially, but not reliably enough to pass. A 12-patch fit passes four of six groups; both held-note groups fail. | [Trainable-input fit](TRAINABLE_BASIC_FIT_RESULTS.md) |
| Does concentrating on three patches solve shared learning? | No. Sound-only reduces their mean errors by 27–46%, but individual requirements fail and original-32 MRSTFT worsens 9.85%. | [Shared comparison](SHARED_BASIC_AUXILIARY_RESULTS.md) |
| Has timbre been isolated from performance? | No. Controlled pitch/duration readouts and changed patch predictions show substantial performance information/sensitivity. A readable factor is not itself proof of harmful reliance. | [Representation probes](BASIC_REPRESENTATION_PROBES_RESULTS.md) |
| Does the tested removal loss fix that? | No useful benefit demonstrated: only 0.235% lower combined reconstruction error than sound-only, with mixed probe changes. Stop this setting, not the broader goal. | [Shared comparison](SHARED_BASIC_AUXILIARY_RESULTS.md) |
| Is loudness the whole held-note problem? | No. Matching still leaves the sound model 7.30%/9.50% worse than its control on those groups. The five unusable unmatched listening comparisons remain confounded. | [Level diagnosis](BASIC_LEVEL_DIAGNOSIS_RESULTS.md), [listening correction](LOUDNESS_MATCHED_LISTENING_RESULTS.md) |
| Does the basic data test velocity separation? | No. The fixed template has no velocity response; low/high-velocity waveforms were identical. | [Controlled probes](BASIC_REPRESENTATION_PROBES_RESULTS.md) |
| Is the balanced-data fit a success? | Unknown. It was at 10/48 updates, nine accepted, when this review snapshot was made. Prefix recovery is not endpoint quality. | [Frozen execution](BALANCED_BASIC_FIT_EXECUTION.md) |

These are development results, often on reused training patches. No current spectral-trained model has passed the complete A2/A3 requirements. The 192 reserved confirmation roots and final-test reservations remain unused. A working CLI and exact replay do not change that conclusion.

## Why the native experiments take hours

The 12,025-render auxiliary comparison took **6 h 15 min**, not five hours. All requests were accepted; the log contains 12,025 attempt rows and 12,025 outcome rows, not 24,050 renders. It made 48 scheduled updates per arm: about **125 requests per scheduled network update**, including evaluation. Numerical gradients, multiple playback observations and measured acceptance amplify the number of calls.

| Recorded portion | Time | Interpretation |
|---|---:|---|
| Full backend calls | 3 h 42 min | Includes plugin creation, state conversion/loading/readback and synthesis. It is not a pure DSP timer. |
| Worker time outside those calls | 2 h 17 min | Includes startup, factory work, QA and communication. Not all of this can be attributed to process spawning. |
| Rest of native execution | 16 min 30 s | Other loop/scoring/checkpoint work outside the timed worker calls. |

The output clips average 4.05 seconds, totaling 13.52 hours of audio. Thus this is already offline rendering, but that does not excuse the overhead. The training session explicitly chooses `max_jobs=1`, creating a process for every request, and the backend creates a plugin for every call. [Worker reuse was qualified in PR 49](RENDER_WORKER_RESULTS.md) with a 16.7% whole-call reduction on its earlier fixtures, yet the later native training path did not adopt it.

The new twelve-request diagnostic reproduces all source waveforms exactly in both modes. Its sequential six-call blocks give 14.26 seconds fresh versus 16.08 reused while another fit runs. That is **not a controlled speedup result**. It retains fresh plugins in both modes, and the Python profile does not isolate all native synthesis time. We need explicit backend stage timers and an interleaved comparison before promising a faster path. The previous description of `render_seconds` as engine-only speed was inaccurate.

## Lessons and decisions

**Make sound quality the organizing objective from the first learning milestone.** Valid serialization, useful derivatives, individual optimization, shared learning and independent quality are separate claims. Advance only on the relevant one. Do not let a long completion ledger turn infrastructure completion into model completion.

**Cross performance factors with patch identity before adding a separation loss.** A previous sampler perfectly associated note pattern with filter enable; later wider inputs repaired that association, but a focused three-patch subset again tied its held-note pitches to patch identity. The existing balanced bank is a useful correction, not proof that data alone explains every failure. Train against the input and changed performances, with event coverage sufficient for attack and release.

**Preserve relevant information rather than enforce perfect statistical independence.** Pitch, envelope response and synthesis can interact. The useful requirement is sound reconstruction under new performances, not chance-level note probes at any cost. Retain patch-information checks and functional audio tests alongside probes. Do not infer separation from branch names or a nonzero auxiliary gradient.

**Measure the cost of learning, not just rendering or backpropagation in isolation.** Report requests per accepted update and unique input coverage per patch alongside end-to-end time. A handful of expensive updates cannot establish that a larger architecture or a longer run will solve the problem. Reuse the successful individual fitting results as diagnostic/possible teacher evidence; do not assume full native finite differences are the scalable final trainer.

**Use cheap learning controls before another native marathon.** If the balanced comparison fails, test whether the same small predictor can learn canonical valid controls from that same balanced feature bank using inexpensive supervised updates. Render its hard outputs to assess the resulting sounds. This isolates learnability of the data/features/model; it is not a change to the final audio-quality objective or permission to rerun the old 404-state initializer unchanged. If that control learns but native sound training does not, investigate the feedback/exposure mechanism before growing the encoder. If native costs remain impractical, qualify optimization-and-distillation on the same scope before retrying a failed proxy or inventing a large model.

**Separate recording-level handling from tone and patch-level accuracy.** Use loudness-matched listening for tone and a separate original-level assessment. Input normalization is a later experimental axis: compare unnormalized, LUFS and peak preprocessing with a fixed gain per full clip. None is selected now. Preserve the gain and keep raw real-render loss; an explicit standardized-level decoder would instead need a verified gain-restoration/export contract. Never independently normalize prediction and target in the primary loss and call level solved. See the [updated plan](../../modelling/TRAINING_PLAN.md).

**Keep the experiments bounded without multiplying the scaffolding.** Maintain source identities, fixed-phase checks, checkpoints and immutable failed results. Reuse the existing runner/audit where it fits; do not add a new experiment framework, micro-PR or multi-agent layer for each small variation. A documented failure should narrow the next decision, not automatically trigger another coefficient or adapter variant.

## Immediate reset

Finish and audit the already-running balanced comparison under its unchanged card. Make no automatic follow-up launch or deadline extension. Profile and fix the repeated-call cost before another long fit. Then establish affordable shared learning on balanced small data, move to held-out patches/performances, and only after independent quality add one useful synthesis capability. The [updated plan](../../modelling/TRAINING_PLAN.md) sets this order; historical cards and scores stay unchanged.
