# Basic reconstruction: spectral-training protocol

Revised 2026-10-31. This specifies the immediate work in the [training plan](../modelling/TRAINING_PLAN.md): establish useful spectral-loss gradients on the small vocabulary, then train and validate the predictor with them. The plan and this protocol are proposals until their implementation and qualification gates pass. Broad preset support remains infrastructure; no broad training or curve-loss tuning is scheduled.

## Prediction contract

Use the existing `PilotCodec`, including its bounds, fixed template, activity masks and identities. Predict oscillator level/pan/unison detune, filter cutoff/resonance, envelope attack/decay/sustain/release, unison voices (1/2/4) and filter on/off. This is nine continuous outputs and categorical sizes three and two: **14 output values**. The original baseline has 355,806 trainable parameters; the current trainable-input comparison has 613,788 total and 257,982 trainable parameters, with the same output count.

The basic model has no learned assets, LFO curves, route slots, mapping curves, custom oscillator curves or seed bits. Other source controls retain the existing template values. The init asset and default patch are part of the saved codec contract. No parameter family is silently removed from complex source audio. This is one oscillator, not a final three-oscillator specification. At export, the five category scores reduce to two integer choices: nine scalar controls plus two selections enter the codec.

`train --prediction-scope basic` checks the exact basic codec before initializing features, training or rendering. Supplying a broad pool fails. The scope is saved in the training contract; changing it on resume fails. Legacy checkpoints retain their original contract under the default `codec` option. Verified same-pool feature seeding can start a new explicitly basic run with fresh learning state; it does not relabel a historical checkpoint.

For a new installation, the existing commands can reproduce the small **parameter-supervised** plumbing check. They do not enable the proposed spectral-loss mode:

```powershell
python research/training/run.py pilot --count 256 --seed 0 --output research/training/outputs/basic-pool
python research/training/run.py train --prediction-scope basic --pool research/training/outputs/basic-pool/pool.json --output research/training/outputs/basic-fit --steps 800 --replay-capacity 32 --batch-size 16 --render-budget 64 --checkpoint-every 100 --cpu-threads 2
```

This is a reproducibility recipe, not an instruction to repeat a completed experiment. Locally, reuse the existing pools/checkpoints below. The historical tiny-fit gate remains an implementation diagnostic; stage advancement uses rendered sound and listening evidence.

## Existing baseline and completed diagnosis

Retain the selected seed-0 checkpoint at `outputs/state-coverage-expanded-16000-20261013/checkpoint.pt` and its 404-training-state pool `outputs/authored-expanded-20261013/pool.json`. Its inference bundle remains `outputs/state-coverage-validation-20261013/selected-model.pt`. Use the original 205-state retrieval checkpoint pinned in the historical reports. Do not change the incumbent by selecting again across old checkpoints.

The [evidence audit](reports/BASIC_RECONSTRUCTION_AUDIT.md) reuses the published 21- and 23-patch development results. Its machine-readable report records input hashes and selects five diagnostic cases at the minimum, lower quartile, median, upper quartile and maximum of the additional panel's existing sound error, with deterministic tie handling. This selection is fixed before any new intervention render. It samples a range of errors, not just the worst examples.

The native diagnostic below is [complete](reports/BASIC_SOUND_DIAGNOSIS.md): 70 accepted requests, 10 exact repeat pairs and no training updates. The unison/stereo correction improves all five cases. The updated priority is to make these differences visible to spectral training and its gradient qualification; a standalone input-feature comparison is no longer the next scheduled experiment. Listening and new-model confirmation remain pending. The original pre-render protocol follows:

- For each selected case, retrieve the original patch and saved baseline prediction by the recorded identities. Verify the historical checkpoint, case and prediction receipts.
- Under each recorded reference/transfer performance, render source, source repeat, baseline prediction, and four separate diagnostic corrections. Each correction replaces one control group with its source values: amplitude envelope; oscillator level; filter enable/cutoff/resonance; or unison voices/detune/pan. Canonicalize inactive values and validate the resulting patch.
- Fix all three oscillator random-phase controls and sampler random phase for every diagnostic render. Preserve other phase controls and seeds. Record these controlled copies separately; use a fresh native process for each request. This is an explicitly controlled diagnosis, not a replacement of historical task scores.
- Score each correction against the same controlled source and baseline. Compare spectral and level/envelope/stereo errors, retain all failures and prepare representative listening clips. A source-value intervention uses oracle information and is never a deployable prediction or a new model-quality result.

The bounded diagnostic is **5 cases × 2 performances × 7 methods = 70 requests**, zero optimizer updates, one worker, at most 8 GiB process-tree memory and a 15-minute wall-time cap. Record a stopped/incomplete run rather than silently increasing the budget. No retries are automatic. An explicit recovery can preserve completed requests and log any replacement attempt. Exact repeats diagnose pipeline correctness; an error is investigated rather than averaged away.

These group interventions locate useful directions but can interact and are not an additive decomposition. If no group helps, inspect observations, representation limits and the metric instead of assuming a parameter loss needs more weight. A large parameter error alone is insufficient evidence to choose a training change.

## A0: contract and correctness

The [local contract check](reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md) now passes: unpooled spectral targets/loss, differentiable basic controls, sound-only gradient connectivity, hard export and Gumbel/optimizer recovery; 33 focused tests, 12 accepted native renders and six exact repeats. Full target replay fits only 62 four-second or 15 sixteen-second examples within 512 MiB. The remaining A0 handoff is the frozen A1 helper experiment card. No qualified helper or spectral-trained inverse model is claimed.

Publish the following before launching a helper or inverse comparison:

| Contract | Required contents and check |
|---|---|
| Scope and data | Exact `PilotCodec`, default patch, bounds, source/lineage/split identities, actual MIDI recipes, observations and source/prediction phase policy |
| Spectral measurements | 44.1 kHz initial profile; FFT sizes 512/2048/8192, proposed quarter-window hops and Hann windows; exact centering, padding, time masks, amplitude convention, silence floor and representation size |
| Objective | Equal-resolution starting combination of spectral convergence and log-magnitude error, plus declared envelope/stereo terms; exact normalization and weights fixed before scoring |
| Differentiable controls | Nine bounded scalar inputs and two one-hot tensors of widths three/two; differentiable inactive-control gates and hard-choice agreement with export defaults |
| Category policy | Straight-through Gumbel-softmax during inverse training; proposed temperature 1.0 to 0.5 over the fixed update schedule; saved RNG; deterministic raw-score `argmax` for native scoring/export |
| Learning state | Helper identity, frozen feature extractor, objective/adapter versions, optimizer/scheduler, auxiliary parameter-loss schedule, replay, sampler and Gumbel RNG, render cursor and runtime identity |
| Resources | Actual per-example/peak bytes, batch size, replay limits, render/update throughput and all counters needed to enforce the plan's caps |

The measurement choices and temperature schedule above are initial proposals, not already implemented defaults. A0 must freeze exact values and formulas in a machine-readable card before A1; justified changes during profiling are recorded there. Inverse input features and spectral training targets are distinct contracts. Old pooled inputs cannot substitute for missing target spectra.

Use nonnegative temporal spectral targets, preserving absolute level and left/right channels. Define envelope targets from windowed RMS and stereo targets that retain side energy; explicitly test centered/one-sided/quiet signals. Any pooling, time chunks or frequency reduction must be recorded and checked on real basic renders for sensitivity to level, pan, detune, filter and envelope changes. Do not assume the previous 24-band/eight-time-bin helper targets are adequate. Report errors in the new target representation and the existing native spectral metric separately.

Keep the helper interface tensor-valued. Hard categories may be represented by one-hot vectors, but converting them to indices for a gather/Python branch can lose their gradient. Use tensor operations for category-dependent defaults and masks. Check codec equivalence under all six basic category combinations, including inactive filter and single-voice detune behavior. Future scalar transpose needs a separately declared rounding approximation in sound training; no new transpose head is added in A0.

Required correctness checks:

- With the parameter loss disabled, the reconstruction loss supplies finite, nonzero gradients to each active continuous family and both categorical heads on deliberately sensitive fixtures. This establishes connectivity only; A1 establishes usefulness.
- Frozen helper and Basic Pitch parameters/buffers do not change; helper input gradients remain enabled. No JSON, `.item()`, detached controls or `no_grad()` on that path.
- Hard training choices decode to the same valid patch as the production adapter; production output is deterministic and has no Gumbel noise. Masks/defaults agree with native inactive behavior.
- Identical-prediction loss is zero within declared numerical tolerance; near-silence stays finite; time padding does not count as signal. Stereo/level changes remain observable.
- Same-runtime interruption/resume reproduces the next selections, minibatches and update with retained tensors. Changing the helper, phase policy, loss or adapter rejects resume. Device migration uses explicit numerical checks, not an exactness claim.

The first spectral-training card fixes oscillator 1–3 and sampler random phase, preserves other phase/seeds, and hashes controlled copies while retaining original split ancestry. Rerender the historical baseline and all candidates under the same policy. Do not overwrite the old codec, source pools, scores or checkpoints to make them appear compatible.

A0 has no model-quality claim. Its cap is 96 native requests, 500 debugging updates and 30 minutes under 8 GiB; checks should use fewer when sufficient. The planning task itself runs none of these experiments.

## A1: qualify the forward model

The [frozen local helper card](reports/BASIC_SPECTRAL_HELPER_CARD.md) implements this protocol with 1,024 training requests and 240 planned qualification requests. Its ten renders per state/performance include isolated pan and detune checks, with 16 recovery attempts reserved inside the 256 qualification cap. Seven focused tests pass. The [completed native result](reports/BASIC_SPECTRAL_HELPER_RESULTS.md) passes forward prediction but fails useful-gradient qualification; it does not permit A2 to begin.

The [aligned-pair correction](reports/PAIRED_SPECTRAL_HELPER_RESULTS.md) is also complete and fails the unchanged gates. It improves 31/48 guided trials, but continuous rankings and transfer detune/joint guidance remain inadequate. Saved-pair inspection shows poor attack/decay/release response fit even on training examples. Check temporal coverage and declare a bounded native response investigation before choosing another full fit or an alternative mechanism; do not expand the prediction vocabulary.

That [bounded time-coverage investigation](reports/SPECTRAL_TIME_COVERAGE_RESULTS.md) is complete: 36 accepted native requests, 12 exact repeat pairs, exact historical audio/feature reproduction and zero optimizer updates. It exposes missed short responses in the sparse training measurements. Preserve envelope coverage and include spectral frames around note starts/ends, then verify signal retention and memory before any further fit. The existing helper also responds poorly under full-recording evaluation; the sampling finding is not a claim of solved reconstruction.

The [sampling repair's native preflight](reports/EVENT_SPECTRAL_FRAMES_RESULTS.md) passes: 32 spectral frames with note-event anchors, complete envelopes, identical pair times, all tested response signals retained and bounded local storage. Twenty focused tests pass. A separately frozen short real-data response fit comes next; the preflight itself performs no learning and does not qualify an artifact.

The [short matched fit](reports/EVENT_RESPONSE_FIT_RESULTS.md) subsequently fails every response-fit group despite improved absolute prediction. Retain the corrected measurements but suspend unchanged full helper fits. The plan's native-optimization alternative now needs its own fixed request budget, numerical-gradient correctness checks and real-audio update qualification before any inverse-training comparison. It must not bypass the small prediction scope or final quality gates.

The [old alpha helper](reports/SOUND_PROXY_RESULTS.md) failed: useful input gradients cannot be inferred from its familiar-performance regression accuracy. Reuse collection, recovery and artifact-refusal behavior. Train a new basic-scope helper with temporal spectral targets and broader training performance coverage; the cause of the previous failure is not assumed to be settled.

The card must fix training families, hard-category balance, continuous perturbations, register/velocity/duration coverage, model size, optimizer, updates and qualification recipes before collection. Use train-only records to fit any scaling statistics. Qualification and transfer recipes must not enter helper training. After a qualification panel guides revisions, label it development and use independent confirmation for the final candidate.

Use a compact temporal model within the initial two-million-parameter ceiling; choose and record its exact layout after the A0 memory profile. No waveform generator or full-Vital emulator is required: the helper predicts only the declared measurements. Collect on demand into a replay bounded by 512 records and 512 MiB, with optional checksummed feature archives capped at 2 GiB. These limits are ceilings, not requirements to retain that much.

Before connecting the helper to the inverse model, test:

1. Prediction on unseen basic patches and held-out performances versus training mean, nearest-record and interpolation forward baselines.
2. Native ordering of nearby continuous changes and hard category alternatives, especially unison/detuning and filter on/off. Predicted rankings must agree with actual sound error, not parameter distance.
3. Short bounded continuous-only and joint continuous/category updates from a fixed imperfect starting patch. The update sees target spectra and evaluator MIDI, never source parameter labels. Evaluate the resulting hard patch through Vital.

Initial working gates, frozen in the card: at least 10% lower helper error than the best simple forward baseline on reference, no worse on transfer; at least 70% native-ranking agreement separately for continuous and categorical comparisons; at least 60% of guided cases improve real error beyond repeat noise and a negative mean change, separately on reference and transfer. Check both continuous-only and joint updates. Report paired family-level intervals and ties; tie tolerance comes from fixed-phase repeats. Reserve minimum comparison counts and failure handling before seeing results. These are feasibility margins, not listening thresholds or release claims.

The cap is 2,048 training requests plus 256 qualification requests, 12,000 helper updates and four hours. Reserve recovery requests inside these caps. A possible qualification layout is 16 independent families × two performances × eight requests (source, repeat, starting patch, positive/negative continuous perturbations, categorical alternative, continuous-guided result, joint-guided result); use fewer planned families if recovery slots are reserved. Freeze actual membership/counts and balance both category types before rendering. Report planned and accepted denominators; failed cases never become successes through omission.

Better helper loss with worse native error fails this stage. Keep the helper artifact marked unqualified and refuse ordinary inverse-training loading. Review the failed directions before proposing one bounded correction or a separately budgeted restricted differentiable synth/native optimization alternative. The earlier render-selected target experiment also failed to improve learning; it is not a proven fallback.

The [native-gradient card](reports/NATIVE_GRADIENT_CARD.md) explicitly budgets the direct-measurement alternative: 624 requests/30 minutes/8 GiB RAM/32 MiB outputs. It tests one measured continuous and joint step on each of 24 development cases, retaining the earlier real-improvement threshold separately on reference and transfer. It does not claim a learned forward model or use its self-prediction gates. Neither this implementation nor the failed helper results qualify a sound-training path by themselves.

The [completed native-gradient result](reports/NATIVE_GRADIENT_RESULTS.md) passes those gates: 24/24 continuous and 23/24 joint improvements, all 560 renders accepted and 24 exact repeats. This bounded direct-measurement path is qualified for a costed A2 trial; the learned helpers remain unqualified. Its inverse loss connection passes gradient/recovery tests, but no spectral inverse model has been trained.

That historical result uses immediate MIDI. The first inverse attempt exposed a [native startup transient](reports/NATIVE_PREROLL_RESULTS.md). The corrected 32,768-frame lead preserves every requested attack/tail and passes strict renderer checks. Its [fresh estimator qualification](reports/SETTLED_NATIVE_GRADIENT_RESULTS.md) passes the same quality gates: 22/24 continuous and 21/24 joint improvements, 560 accepted renders and 24 exact repeats. Old targets and the partial model prefix are not corrected evidence. The new runners pin profile/source identity and refuse cross-profile qualification/cache reuse. Fresh A2 starts from the same historical initial model with new optimizers and targets.

## A2: spectral-loss inverse training and native comparison

Only a qualified, hash-pinned helper or native-gradient estimator may guide this comparison. Start the parameter-only control and spectral candidate from the same incumbent checkpoint with fresh, matched optimizer state. Keep the basic encoder/head architecture, source population, phase policy, input preprocessing, training observations and inverse-update count fixed. Report helper training or native-gradient query costs separately; a direct native path must fit all probes into its declared budget.

Freeze helper weights for the full inverse run while retaining derivatives through inputs. Train with the declared spectral/envelope/stereo loss and Gumbel-softmax categorical path. The parameter anchor is optional and must have a fixed scale/decay; report its gradient contribution and show sound-only categorical gradients without running a loss-weight sweep. Qualify the hard-decoded results, not a soft blend of incompatible categories.

Infer one patch from a reference recording; reuse that prediction to match real target spectra from the same source under reference and additional training MIDI. Cover attack, sustain and release with sufficient observation length. MIDI conditions only the helper/evaluator. Include a new approximately 32-example fit check because the objective changed, with its updates/renders counted inside A2; it is not generalization evidence.

The card must specify native audit cadence, candidate selection and stopping rules before training. At audits, export with deterministic `argmax`, render real Vital and compare against both the historical incumbent and matched continued parameter-only control on the same development cases, alongside template/retrieval. Monitor invalid outputs, category usage, helper/native disagreement and reference/transfer errors. Do not promote a model because the helper's training score improved. Stop at predeclared divergence/invalidity limits and retain the last native-checked checkpoint; an altered helper requires a new qualification/comparison.

Cap A2 at 1,024 native requests including fit/audits/recovery, 8,000 inverse updates per arm and four hours total. Profile first; no automatic extension if the spectral candidate is slower. Freeze a practical native gain margin and earlier-stage/tail tolerances before scoring. The working primary improvement target is 10%; exact source-category accuracy is diagnostic, not an acceptance gate.

For the qualified direct-native alternative, the [separate frozen card](reports/NATIVE_INVERSE_FIT_CARD.md) explicitly revises the request allocation to 3,328, with 128 matched updates per arm on 32 training states. The four-hour, 8-GiB RAM and 2-GiB archive limits remain. This is a predeclared replacement for the helper-based request allocation, not an extension of a completed run. Both training-performance fit gates precede the 12-state development comparison; no test states or automatic checkpoint selection are used.

After the renderer correction and requalification, use the [fresh settled-audio card](reports/SETTLED_NATIVE_INVERSE_CARD.md). It preserves the original initial weights, matched design and thresholds, with new target measurements and optimizers. The stopped cold-renderer model prefix is not its initialization.

That [fresh trial is complete](reports/SETTLED_NATIVE_INVERSE_RESULTS.md): both arms take 128 updates and the sound arm passes both fit gates, but development gains of 5.48% reference / 3.50% transfer miss the 10% target. Complete offline replay succeeds, including a preserved checkpoint before a host-sleep timeout. No model is promoted. The category-score learning signal is often tiny despite strong native sound sensitivity; investigate that path before another costed comparison, preserving the small vocabulary and existing acceptance thresholds.

The [fixed-scale diagnosis](reports/CATEGORY_SIGNAL_RESULTS.md) now passes its derivative gates with zero new native requests: training-only constants restore both category-head signals while preserving every checked initial hard patch and the entire historical replay. Integrate the same frozen scales into both loss arms and checkpoint their identity before a fresh matched fit. Production stays at raw-score argmax. This does not change the 14-output vocabulary or satisfy the outstanding sound-quality gates.

## A3: fresh confirmation and first release

After a promising A2 result, freeze the candidate and run a second training seed with a separate card capped like A2. Choose the confirmation population size from development variability and fix the candidate comparison before new confirmation scoring. Do not multiply seeds while the sound-training mechanism remains unqualified.

Both existing panels are development data. For confirmation, append independent authored families with the same basic vocabulary and immutable split inheritance; use only newly assigned validation families. Verify against all existing training/development identities and reserve every test assignment. Freeze the complete validation membership and performances before candidate scoring. The confirmation population size and render budget must be fixed from the observed development variance, not adjusted after seeing a favorable candidate score.

Use template, training-only retrieval and unchanged model controls. Report reference/transfer error separately, failure rates, median/tail behavior and uncertainty by independent family. The working practical target is a 10% primary-error improvement, with a paired interval excluding no benefit and declared earlier-stage/tail regression limits. No claim of stage completion follows solely from parameter accuracy or a lower training loss; the second seed and blinded listening assessment remain required.

Also retain the matched parameter-only control. The [192-family reservation](reports/BASIC_CONFIRMATION_RESERVATION_CARD.md) declares a conditional 3,840-request / four-hour confirmation design, with no optimizer updates. Its complete execution plan and a passing A2 result remain required before collection. Do not stop when results happen to look favorable. Prepare randomized/blinded listening clips with a declared dimension: loudness-matched tone versus separate original-level accuracy. Record actual judgments; automatic scores do not substitute for listening.

The [local listening tool](LISTENING.md) prepares anonymous, counterbalanced reference/A/B packets and audits all planned judgments, including ties/neither. Tone packets default to checked constant-gain LUFS matching; explicit level-accuracy packets preserve original levels. Original recordings and raw sound-quality gates remain unchanged. The [unmatched diagnostic responses](reports/LOUDNESS_MATCHED_LISTENING_RESULTS.md) are loudness-confounded and do not establish a tone winner. A representative population and acceptance rules still need a frozen card after a promising candidate. No synthetic response satisfies the listening gate.

Package a basic inference bundle and CLI with pinned preprocessing, weights, codec/defaults, assets/notices and an audio-only example. Neither helper weights nor true MIDI are inference dependencies. Verify a fresh-environment load, export/native round trip, latency/memory and recovery. Record the model's bounded scope and remaining failures. Final-test use belongs to the frozen release candidate; it is not part of the development comparisons above.

## Work order and status

| Work package | Current state | Reviewable deliverable |
|---|---|---|
| Baseline audit and native diagnosis | Complete; listening still pending | Existing reports, checkpoints and render receipts |
| A0 spectral/control contract | Complete; both local A1 cards frozen | [Versioned contract and measured resource evidence](reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md) |
| A1 useful-gradient qualification | Native path qualified on the corrected renderer; learned helpers refused | [Settled-gradient evidence with all failures retained](reports/SETTLED_NATIVE_GRADIENT_RESULTS.md) |
| A2 spectral inverse comparison | [Focused shared comparison](reports/SHARED_BASIC_AUXILIARY_RESULTS.md) improves the three sounds by 27–46% on average but fails per-input and preservation gates; auxiliary effect is only 0.235% | Balance the same notes and durations across all three sounds; freeze the next sound-training comparison before fitting. Keep 14 outputs and complete six-group/earlier-32 checks before confirmation |
| A3 confirmation/basic bundle | Pending A2; [192-family reservation](reports/BASIC_CONFIRMATION_RESERVATION_CARD.md) prepares fresh data without rendering or scoring | Complete the conditional execution plan, independent confirmation, second seed and listening; fresh-runtime inference now passes |

Use separate reviewable PRs for these work packages and local checks with GitHub Actions disabled. Publish exact run status, resource use and failed gates; do not equate implemented code with a passed training milestone. All upcoming work stays local; Colab notebooks, transfers and Linux qualification are deferred.

Only after basic-stage quality passes should the next PR introduce a bounded new capability. One LFO and at most two linear routes are the initial modulation limit. Optional compact mappings for the first two routes remain deferred; no 64-slot mapping output layer is part of this schedule.
