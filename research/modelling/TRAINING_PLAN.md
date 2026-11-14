# OBRUXO training plan

Revised 14 November 2026 after reviewing the original plans, recorded results and PRs. **Learn to turn one isolated recording into a playable Vital patch that sounds similar and responds sensibly to different notes. Prove this with a small model before adding synthesis complexity.** Exact source settings, graph recovery and exhaustive Vital coverage are not the objective.

The [review and lessons](../training/reports/TRAINING_REVIEW_20261031.md) explain how the work diverged and what the results establish. The [status ledger](../training/IMPLEMENTATION_STATUS.md) separates working infrastructure from unfinished quality. The [experiment protocol](../training/STAGED_RECONSTRUCTION.md) defines how to run the next comparisons. This document determines future work order; existing frozen cards, failed results, deadlines and reserved splits are unchanged. The [previous plan](https://github.com/noisedoctrine/obruxo/blob/ced6b0b/research/modelling/TRAINING_PLAN.md) remains historical evidence.

## What success looks like

Inference takes audio only: no supplied MIDI, source patch, asset labels or optimization oracle. Start with isolated synth/instrument recordings and an explicit supported sound vocabulary; arbitrary mixes and source separation are outside this plan. The original 2–16-second input range is a proposed release contract to validate, not a claim about every existing runner.

Infer one patch from a recording, then compare its real Vital sound with the source under both the observed performance and different pitches, durations and playing patterns. A patch that only matches one held note is not sufficient evidence of recovered instrument character. Evaluate velocity only when the declared patch/template actually responds to it.

A first useful release may support a bounded vocabulary. It needs demonstrable sound quality, representative blinded listening, independent confirmation, an audio-only offline bundle, reproducible local training and measured CPU/iGPU runtime. The earlier 15–20-second inference and roughly 8-GiB application-memory aspirations remain targets to measure, not guarantees. There is no requirement to predict all Vital fields before releasing a useful model.

## Starting point

The renderer, codec, fixed-phase checks, checkpoint recovery, real-audio measurements and basic inference path work in the qualified Windows runtime. Bounded native gradients and individual fitting work: the three difficult direct fits reduce joint error by 41–86%. Shared audio-to-patch learning remains incomplete: held-note failures and earlier-sound regressions persist. The tested pitch-removal setting adds only 0.235% mean improvement and is not adopted. Neither branch naming nor probes establish successful performance/timbre separation.

The balanced three-sound fit from [PR 139](https://github.com/noisedoctrine/obruxo/pull/139) is complete and independently audited. It trained from real-Vital sound feedback across the balanced pitch-and-duration grid, but failed the per-sound, most grid, wider-input and earlier-sound gates; see the [execution record](../training/reports/BALANCED_BASIC_FIT_EXECUTION.md). A separate [48-update source-label diagnostic](../training/reports/BALANCED_SUPERVISED_CONTROL_RESULTS.md) improved setting metrics but worsened average real-Vital sound scores in every source group. The four-schedule [proxy comparison](../training/reports/PRESET_LOSS_SCHEDULE_RESULTS.md) is also complete on a different model and training bank; raw and gain-matched scores favored different schedules. No candidate is admitted, and the held-out reserve remains untouched.

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

### 1. Diagnose the completed sound-fit failure before another fit

The 48-update balanced fit already tested real-Vital sound feedback on the three sounds crossed with MIDI 36/61/83 and short/long notes. It used 6,292 native render attempts over 3 hours 24 minutes. Training gradients reached the active branches, but all three per-sound gates failed; only one of three pitch groups passed, wider alternate-input quality regressed, and earlier-sound preservation failed. The [execution record](../training/reports/BALANCED_BASIC_FIT_EXECUTION.md) and [frozen card](../training/reports/BALANCED_BASIC_FIT_CARD.md) contain the scope and gates. Do not repeat this fit just to check whether audio feedback reaches Vital.

The separate [source-label schedule comparison](../training/reports/PRESET_LOSS_SCHEDULE_RESULTS.md) used a fixed learned audio proxy, a different model and a 172-family training bank. It scored 21 validation sounds through real Vital at one pitch each. Fading led on raw MR-STFT, while the paper-style fixed loss led after gain matching; their direct differences were inconclusive. This does not select a schedule for the balanced native-feedback model. Do not repeat these four schedules.

Next, inspect the already collected fit trace and per-sound/per-performance comparisons to identify why the model improved some groups but failed the full gate. Reuse compatible evidence; do not rerender or repeat a schedule screen. Only after a specific, evidence-backed cause is found should a bounded test change one training factor. Keep the 14 outputs, existing sound/preservation gates, 18 development clips and untouched reserve/test. If retained traces cannot answer the question, state that limitation and freeze a small diagnostic before allocating another long native run.

Any later native fit still needs a fresh timed preflight and checkpointed allocation. The 103-second label-only check is not a forecast for sound feedback.

### 2. Broaden shared learning only after the sound-quality gate passes

Keep the eighteen-clip bank as a controlled learning fixture: the same three source patches crossed with MIDI 36/60/84 and 0.4/1.2-second notes. It is development data, not independent validation.

If a sound-loss model meets the unchanged per-sound, grid and earlier-sound gates, expand to broader patches and held-out performances with the same 14 outputs. If none meets them, inspect feature information and the audio-update path before adding model capacity or Vital controls. Preserve all previous gates; a better average cannot hide a broken source group or earlier sound.

Use the existing audited sound-only balanced fit as the zero-label comparison if its exact inputs, initializer and renderer match the new schedule arms; do not repeat it merely to fill a table. The new arms must start from the same checkpoint, use the same examples/update count and production decoding, and be judged on the same real-Vital clips. Retain label accuracy only as a diagnostic. The [individual direct fits](../training/reports/DIRECT_BASIC_CONTROL_RESULTS.md) are reference solutions, not a deployed audio model. Optimize-and-distill or a revised differentiable helper remains a separate proposal requiring real-audio qualification.

**Exit:** a shared sound-trained model learns every declared small-case/input group, maintains the earlier panel, and has a measured affordable learning path. Existing frozen A2 criteria remain unchanged. Small-bank fitting establishes learnability, not generalization.

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

**2026-11-14 inverse-method screen:** on 21 development families, direct Vital candidate edits and a small local score fit improved gain-matched error in 20/21 cases when supplied the correct note events. Proxy-gradient patch updates and per-clip model tuning were inconclusive. The follow-up used audio-estimated events with no source MIDI, but onset and duration errors confounded the scores. The panel also informed checkpoint/proxy selection, so neither run is independent promotion evidence. See the [screen report](../training/reports/INVERSE_METHOD_SCREEN_RESULTS_20261114.md) for measurements, limits, and the missing reproducibility details.

**Deferred until the base model passes its reconstruction-quality gate:** testing a fixed list of patch edits in Vital, fitting a local curve to Vital scores, adjusting patch values through the neural sound copy at inference, and fine-tuning a temporary model copy for one clip (TTT). These are inference-time add-ons and do not change shared model training. This does not defer a sound proxy that the core model needs for audio-loss gradients during training.

## Execution and reporting rules

All upcoming work stays local/free-first. Colab notebooks, Linux renderer qualification, paid resources and distributed training are deferred. A GPU cannot fix CPU plugin setup or repeated native-query costs by itself. Use the qualified CPU float32 renderer; choose XPU training only after measuring the coupled path. Start with one native worker, two Torch threads and a 4-GiB process-tree planning ceiling for small trials; cards may specify justified measured limits. Do not silently increase concurrency or weaken QA.

Before execution, a short card names the question, changed variable, comparator, exact data/splits, model/output counts, objective, accepted-update accounting, request/time/RAM/storage ceilings and pass/stop/failure rules. Measure the proposed schedule, not just a single clip. Save recipe identities and checkpoints; retain enough fixed audio for exact phase-controlled correctness and independent endpoint rescoring. Distinguish replay of stored training scalars from rescoring retained waveforms. Keep rejected attempts and all charged resources.

Use the existing runner/audit and compact reports where possible. One PR should answer one substantive question or deliver one necessary capability; avoid separate micro-PRs for routine launch/status bookkeeping. Keep PRs ready when reviewable, run appropriate local checks with Actions disabled, and never merge without explicit approval. Report results and what decision they support; do not use PR count, connected gradients or exact recovery as substitutes for learned sound quality.
