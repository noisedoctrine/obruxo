# OBRUXO training plan

Revised 31 October 2026 after reviewing the original plans, recorded results and PRs. **Learn to turn one isolated recording into a playable Vital patch that sounds similar and responds sensibly to different notes. Prove this with a small model before adding synthesis complexity.** Exact source settings, graph recovery and exhaustive Vital coverage are not the objective.

The [review and lessons](../training/reports/TRAINING_REVIEW_20261031.md) explain how the work diverged and what the results establish. The [status ledger](../training/IMPLEMENTATION_STATUS.md) separates working infrastructure from unfinished quality. The [experiment protocol](../training/STAGED_RECONSTRUCTION.md) defines how to run the next comparisons. This document determines future work order; existing frozen cards, failed results, deadlines and reserved splits are unchanged. The [previous plan](https://github.com/noisedoctrine/obruxo/blob/ced6b0b/research/modelling/TRAINING_PLAN.md) remains historical evidence.

## What success looks like

Inference takes audio only: no supplied MIDI, source patch, asset labels or optimization oracle. Start with isolated synth/instrument recordings and an explicit supported sound vocabulary; arbitrary mixes and source separation are outside this plan. The original 2–16-second input range is a proposed release contract to validate, not a claim about every existing runner.

Infer one patch from a recording, then compare its real Vital sound with the source under both the observed performance and different pitches, durations and playing patterns. A patch that only matches one held note is not sufficient evidence of recovered instrument character. Evaluate velocity only when the declared patch/template actually responds to it.

A first useful release may support a bounded vocabulary. It needs demonstrable sound quality, representative blinded listening, independent confirmation, an audio-only offline bundle, reproducible local training and measured CPU/iGPU runtime. The earlier 15–20-second inference and roughly 8-GiB application-memory aspirations remain targets to measure, not guarantees. There is no requirement to predict all Vital fields before releasing a useful model.

## Starting point

The renderer, codec, fixed-phase checks, checkpoint recovery, real-audio measurements and basic inference path work in the qualified Windows runtime. Bounded native gradients and individual fitting work: the three difficult direct fits reduce joint error by 41–86%. Shared audio-to-patch learning remains incomplete: held-note failures and earlier-sound regressions persist. The tested pitch-removal setting adds only 0.235% mean improvement and is not adopted. Neither branch naming nor probes establish successful performance/timbre separation.

The balanced three-sound fit from [PR 139](https://github.com/noisedoctrine/obruxo/pull/139) is already running under its frozen card. Its final result is pending at this revision. Complete and audit that run; do not change its inputs, implementation, objective or deadline. Merging its PR did not complete the experiment. This planning change launches no additional training or confirmation scoring.

## Keep the prediction task small

Use the current one-oscillator basic codec, stage defaults and valid native ranges:

| Component | Predictions | Dense values |
|---|---|---:|
| Oscillator 1 | Level, pan, unison detune | 3 |
| Filter 1 | Cutoff, resonance | 2 |
| Amplitude envelope | Attack, decay, sustain, release | 4 |
| Oscillator 1 unison | Scores for 1, 2 or 4 voices | 3 |
| Filter 1 enable | Scores for off/on | 2 |
| **Total** | **9 continuous values and 2 categorical choices** | **14** |

Production argmax reduces the five category scores to two selections: **11 predicted values enter the codec**. The codec supplies the rest of the patch from its fixed template. The original compact predictor has 355,806 weights; the current correction model has 613,788 total, with 257,982 trainable. These are implemented architectures, not a prescribed final model size. Imported Basic Pitch weights and running statistics stay frozen; the trainable timbre encoder/adapters remain separate.

There are no predicted LFOs, routing slots, mapping curves, custom oscillator curves or random-seed bits in this stage. Comprehensive preset support remains available infrastructure. Do not silently simplify a complex source label while training against audio outside the basic vocabulary. Keep waveform/stack/warp defaults fixed. For later three-oscillator designs, level/pan/detune would require nine values, not three; semitone transpose should use one bounded scalar per oscillator with declared export rounding. Every expansion needs an exact named output inventory before implementation.

## Next work, in decision order

### 1. Close the current comparison and make learning affordable

Publish the balanced fit's complete result and independent audit under the [unchanged card](../training/reports/BALANCED_BASIC_FIT_CARD.md), including every failed group, earlier-32 preservation, accepted updates and full costs. A timeout or deadline is a recorded incomplete experiment, not permission to extend automatically. Do not repeat a control already retained with compatible inputs and renderer identity.

Before another long native fit, profile the complete basic training path. The recent auxiliary comparison spent 6 h 15 min on 12,025 requests for only 48 updates per arm. About 2 h 17 min was worker time outside the backend call; backend timing also includes plugin creation and state checks. Pure synthesis speed is still unmeasured. [Detailed accounting](../training/reports/TRAINING_REVIEW_20261031.md#why-the-native-experiments-take-hours)

Add stage timings for process startup, plugin creation, patch conversion/loading/readback, synthesis, QA/transfer, features, optimizer and checkpointing. Measure cold and warm calls, requests per accepted update and unique input coverage. Run interleaved comparisons with no competing fit before drawing speed conclusions. Start by qualifying the existing bounded worker reuse on the current basic profile. Consider plugin reuse or cached invariant setup only if timings justify them and A–B–A state-reset, polyphony/tail, fixed-phase and failure-restart checks pass. Keep required correctness; avoid redoing invariant work per numerical probe when a verified equivalent path exists.

**Exit:** a verified local execution policy and a realistic forecast for the actual update schedule. Aim for roughly 30-minute learning diagnoses; this is a planning target, not a speed result or retroactive cap on the current run. A longer pilot requires an explicit useful question, measured forecast and fixed cap. If native calls still dominate, change how feedback is obtained rather than buy another long run with very few updates.

### 2. Prove shared learning on balanced small data

Use the current eighteen-clip bank: the same three source patches crossed with MIDI 36/60/84 and 0.4/1.2-second notes. All patches receive the same performance coverage. Preserve the old evaluation cases and earlier-sound panel; new grid cells supplement them. This bank is a controlled learning fixture, not independent validation or the final dataset.

If the running candidate passes its full gates, retain it and proceed to wider learning rather than run another mechanism experiment. If it fails, first ask a cheap, decisive question: **can this model learn valid controls from these same balanced inputs at all?** Use a bounded supervised control with the existing features and canonical codec targets, enough cheap updates to expose a fitting curve, and real-render endpoints. This is a new small learnability diagnostic, not a rerun of the old 404-state initialization. Exact parameter error is diagnostic; the exported patch's sound remains decisive.

| Result | Next decision |
|---|---|
| Shared control cannot fit the small balanced bank | Inspect feature information, temporal coverage and model optimization. Compare at most one simpler/direct trainable predictor against the existing correction model before further native sound fitting. |
| Shared control fits; native sound training still fails | Investigate feedback quality and useful update exposure. Local derivative success does not establish effective network training. Do not add capacity or separation losses without evidence. |
| Native cost prevents adequate learning exposure | Test amortizing successful native optimization into teacher targets and inexpensive shared-model training on this same scope. Qualify teacher sounds across the performance grid first. This is an alternative to repeated full finite differences, not a proven solution. |
| Shared sound model fits and preserves earlier cases | Move to wider patches and held-out performances with the same 14 outputs. |

Do not restart the failed helper unchanged, sweep adversary coefficients, add another frozen-base adapter by default, or run a seed grid here. The [individual direct fits](../training/reports/DIRECT_BASIC_CONTROL_RESULTS.md) are useful reference solutions, not a deployed audio model. Optimize-and-distill or a revised differentiable helper needs its own bounded real-audio qualification. Source labels may initialize or diagnose; they do not replace the perceptual objective.

**Exit:** one shared audio-only model learns every declared small-case/input group, maintains the earlier panel, and has a measured affordable learning path. Existing frozen A2 criteria cannot be bypassed by calling a failed result a new milestone. Small fitting establishes learnability, not generalization.

### 3. Establish basic generalization before more controls

Expand from the three-patch fixture to the existing twelve-/approximately 32-patch learning checks, then a declared wider training population. Reuse valid compatible records; generate further training audio on demand from recipes with a bounded queue/replay. A fully pre-rendered corpus is not required. Select population and number of updates from measured learning curves and throughput, rather than the old 3,000/15,000/60,000-state targets.

Cross or counterbalance pitch, duration and playing pattern independently of patch category. Ensure held-note coverage per patch and include informative onsets/releases. Reserve performances and entire source families separately; a new note from a training patch is performance transfer, not unseen-patch validation. Real velocity responsiveness is a later declared template/capability change; the current silent velocity contrast cannot qualify invariance.

Infer from multiple individual recordings during training, and evaluate each inferred patch under its own and other performances. Keep sound reconstruction primary, with explicit rehearsal of earlier cases. If a parameter anchor is used, declare its scale and actual contribution. Use production hard argmax in the forward/native/export path with the qualified approximate backward derivative; Gumbel-softmax remains a historical/alternative adapter requiring a fresh matched qualification. Neither makes Vital differentiable.

Report fixed-template, train-only retrieval, matched settings-trained control and the strongest compatible earlier sound model. Report combined loss and MRSTFT separately, per input type and family, with envelope, level, stereo, invalidity and tail diagnostics. Native loss remains measured on real Vital, not solely a helper prediction. Keep the existing six-group and earlier-32 requirements for their frozen comparisons.

**Exit:** meaningful real-render improvement on all declared reference/transfer strata, rather than an average that hides a broken register, with no undeclared earlier-sound regression. The inherited working development margin is 10%; exact endpoints, comparator, tolerances and stopping rules must be frozen in the applicable card. A passing development candidate then proceeds to independent confirmation.

### 4. Confirm one bounded model and deliver it

Freeze the candidate and reproduce the promising learning change on a second seed under a separately fixed budget. Use the 192 reserved independent authored roots only after A2 qualifies and their complete conditional execution plan is frozen. Their reservation is not permission to score them during diagnosis. Preserve original test assignments and ancestry; both earlier basic validation panels are development data now.

Confirm improvement under reference and held-out performances, with family-level uncertainty, failure/tail reporting and earlier-stage preservation. Retain the working 10% primary improvement target and an interval excluding no benefit, subject to the applicable predeclared design. Compare the frozen candidate with the same meaningful controls; do not select favorable subgroups afterward.

Run representative randomized/blinded human comparisons. Tone comparisons use checked constant-gain loudness matching; original level accuracy is assessed separately. The first twelve responses had five severe loudness confounds and establish no clean tone winner. The corrected packet is available but no new judgments are recorded. Actual listening is required; neither a metric nor simulated judgments satisfy it.

Complete weights, pinned preprocessing, decoder/defaults, asset/notices, local audio-only CLI, model card, documented limitations and fresh-environment reproduction. Existing basic inference/runtime work is reusable, not a finished-model quality claim. Freeze release scope and policy before the untouched final test; using test results to revise the model consumes that test's independence.

### 5. Add one useful capability at a time

After basic quality passes, choose an expansion from measured failures: a small factory-waveform vocabulary, filter envelope or second oscillator, for example. Measure the old model's representation limit on the new sounds, then whether the added capability closes it without harming earlier cases. A third oscillator follows evidence of benefit, not codec availability.

For modulation, begin with **one LFO and at most two linear routes**, one source and a short explicit destination list. Start with simple shapes/rate/phase and enough audio to observe their motion. Evaluate new note lengths, tempo and retrigger behavior. Different valid graphs with similar sound need not match the exact source graph. The full codec's indexed dependencies still forbid arbitrary route sorting.

Effects, sampler, additional envelopes/oscillators and more complex modulation follow the same rule. Custom mappings remain optional late work for only the first two routes with a proposed ceiling of six ordered knots each; custom oscillator curves are also optional. Eight simultaneous complex LFOs, all 64 routes/maps and exact seed recovery have no scheduled prediction milestone. Before each addition, count its continuous values, categorical logits, total model weights, render cost and expected audible benefit; requalify the training feedback for that scope.

## Deferred research choices

**Input normalization:** compare unnormalized audio, per-clip LUFS normalization and per-clip peak normalization later, on identical data/model/budgets. No method is chosen and this is not a prerequisite for the current fit. Apply one gain to both channels over the whole clip; retain gain/meter/headroom information and handle silence explicitly. LUFS and peak scaling can change which information is easy to learn; neither guarantees pitch/timbre separation.

For the first such comparison, normalize only model input/features, preserve the original target and calculate raw real-render reconstruction loss. No reversal is needed on this loss path. Preserve the normalization gain as versioned input metadata so absolute level is not unknowingly discarded; this adds no prediction head. An alternative standardized-level prediction must restore the recorded gain **before loss and in the exported playable patch**, with verified legal range and native equivalence. External post-render scaling alone is not a complete patch-export solution. Gain cannot be moved across nonlinear synthesis/effects without proof. Independently normalizing prediction and target would intentionally remove level error and is a different objective.

**Representation losses and architectures:** retain controlled probes and patch-information checks, but judge interventions by reconstruction under changed performances. Do not require complete removal of note information when it is useful to explain synthesis response. Revisit separation losses, cross-attention, JEPA, large encoders or generative decoders only when a specific unresolved failure and a cheap comparison justify them. Earlier negative screens are not universal impossibility results, but they are reasons not to repeat the same setting.

## Execution and reporting rules

All upcoming work stays local/free-first. Colab notebooks, Linux renderer qualification, paid resources and distributed training are deferred. A GPU cannot fix CPU plugin setup or repeated native-query costs by itself. Use the qualified CPU float32 renderer; choose XPU training only after measuring the coupled path. Start with one native worker, two Torch threads and a 4-GiB process-tree planning ceiling for small trials; cards may specify justified measured limits. Do not silently increase concurrency or weaken QA.

Before execution, a short card names the question, changed variable, comparator, exact data/splits, model/output counts, objective, accepted-update accounting, request/time/RAM/storage ceilings and pass/stop/failure rules. Measure the proposed schedule, not just a single clip. Save recipe identities and checkpoints; retain enough fixed audio for exact phase-controlled correctness and independent endpoint rescoring. Distinguish replay of stored training scalars from rescoring retained waveforms. Keep rejected attempts and all charged resources.

Use the existing runner/audit and compact reports where possible. One PR should answer one substantive question or deliver one necessary capability; avoid separate micro-PRs for routine launch/status bookkeeping. Keep PRs ready when reviewable, run appropriate local checks with Actions disabled, and never merge without explicit approval. Report results and what decision they support; do not use PR count, connected gradients or exact recovery as substitutes for learned sound quality.
