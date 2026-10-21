# OBRUXO training plan: learn to reconstruct sound

Revised 2026-10-21. **Train using perceptual spectral reconstruction loss on a small prediction space, prove improvement in real Vital audio, then add capabilities when they improve audible results.** Parameter matching is an initialization or auxiliary signal. Comprehensive preset serialization does not determine the model's output size. This revision moves sound-based learning into the first stage and supersedes the [coverage-first snapshot](TRAINING_PLAN_COVERAGE_20261018.md), including its sequence of expanding coverage before establishing useful reconstruction gradients.

The [completion ledger](../training/IMPLEMENTATION_STATUS.md) tracks the active work. The [basic-stage protocol](../training/STAGED_RECONSTRUCTION.md) fixes the immediate implementation and qualification steps. Prior code, checkpoints, reports and split reservations remain available. This is an updated plan, not a claim that spectral-loss training is implemented or qualified.

## Target and current evidence

One isolated instrument recording enters; one valid, playable `.vital` patch leaves. Inference receives audio only. The useful result is a similar sound, including its response to new notes, durations and velocities. Recovering the exact source settings or routing graph is not the product goal.

The small problem has partial evidence already: the nine-control/two-choice pilot learned from 205 and later 404 training patches. Its [original validation](../training/reports/STATE_COVERAGE_RESULTS.md) and [additional validation](../training/reports/FRESH_VALIDATION_RESULTS.md) show reconstruction gains, but persistent envelope, level and unison errors. Neither proves reliably good reconstruction or listener usefulness. Both panels (21 and 23 patches) have been used for development. They must not be relabelled as new confirmation data.

The [native basic diagnosis](../training/reports/BASIC_SOUND_DIAGNOSIS.md) is complete: 70 accepted renders and 10 exact phase-fixed repeat pairs. Correcting unison voices/detune/pan reduces mean error by 26.68% across five development cases. This uses known source controls and is not a trained-model gain. Stereo and detuning must therefore remain visible to the sound objective and qualification checks.

An [earlier sound-prediction helper](../training/reports/SOUND_PROXY_RESULTS.md) on the broader alpha vocabulary failed qualification: its guided updates lowered predicted loss on all 20 cases but increased real Vital error on all 20. Transfer prediction also failed. The collection/recovery code is reusable; that helper is not qualified to train the predictor. The [direct-render target experiment](../training/reports/RENDER_SELECTED_RESULTS.md) also failed to improve inverse learning, increasing held-out error by 12.56%. Neither route is an established solution merely because its plumbing exists.

The latest eight-example full-model run remains an infrastructure diagnostic. Further all-output fitting, broad route coverage and eight-LFO loss tuning are deferred. The next substantial milestone is useful sound-loss gradients on the basic vocabulary, followed by a real-render improvement in the predictor.

## Separate preset support from prediction scope

Keep the comprehensive reader, serializer, native readback checks, source provenance and on-demand renderer. They allow us to inspect and reproduce complex patches. A training stage has its own explicit output vocabulary and complete default patch. Its model allocates heads only for that vocabulary.

Unused features are fixed in the stage's authored sources and decoded predictions. Do not feed a complex original preset's audio to the basic stage with a silently simplified label. Future training on sounds outside a stage's representable vocabulary needs an explicit approximation objective and separately reported representation error.

| Capability | Current training decision | Later admission rule |
|---|---|---|
| Basic oscillator/filter/envelope controls | Use the existing nine continuous controls and two categorical choices | Establish rendered reconstruction quality first |
| Factory waveforms, second oscillator, extra envelope or effect | Deferred from the immediate baseline | Add one bounded capability to address a measured sound failure |
| LFOs | No LFO heads in the basic model | Start with one LFO, a small shape vocabulary, rate and phase; introduce a complex-curve representation only if simple shapes are limiting |
| Modulation routing | No route heads in the basic model | At most two ordered routes initially, one LFO source and a short named destination list; add other sources only when needed |
| Modulation mapping curves | Linear default, no predicted maps | Exceptional later trial: only the first two routes, at most six ordered knots per map; validate its audible benefit before adopting it |
| Custom oscillator curves | No predicted custom curves | Test a compact representation for one oscillator only if a reconstruction failure requires it |
| Random seeds | Fixed configuration for the basic stage, no seed prediction heads | Treat stochastic variation as a nuisance or sampling policy; exact seed-bit recovery is not a training or release objective |
| Full 64-route/eight-LFO/custom-curve model | Preserved as experimental infrastructure | No automatic place in the curriculum and no exhaustive-output release requirement |

The six-knot map limit is a proposed ceiling, not an implemented or qualified representation. A nonlinear route map can differ from changing its source LFO, especially when several routes share that LFO; the current decision is to exclude this complexity, not assert universal equivalence.

A reduced model must actually have fewer output heads. Masking a 64-route decoder's labels does not satisfy this requirement. Record continuous values, categorical heads, categorical logits, trainable parameters and runtime for each proposed expansion. The basic model has **9 continuous outputs + 5 categorical logits = 14 output values**, with **355,806 trainable parameters**. It does not instantiate the broad model and mask most of it.

| Stage A component | Predicted controls | Dense output values |
|---|---|---:|
| Oscillator 1 | Level, pan, unison detune | 3 |
| Filter 1 | Cutoff, resonance | 2 |
| Amplitude envelope | Attack, decay, sustain, release | 4 |
| Oscillator 1 unison count | Scores for 1, 2 or 4 voices | 3 |
| Filter 1 enable | Scores for off/on | 2 |
| **Total** | **9 continuous controls and 2 category selections** | **14** |

At export the five category scores become two integer selections: **11 predicted control values enter the existing codec**, which fills the remaining patch from its fixed template. The differentiable path retains nine scalar values and five selection-tensor entries. These are one-oscillator Stage A counts, not the eventual three-oscillator model. The helper's spectral outputs are training-only and do not enlarge the exported patch head.

For future expansions, semitone transpose is one bounded continuous prediction, rounded at export. Three oscillator transposes need **3 values instead of 291 scores**, saving 288; three oscillators' level/pan/detune need **9 values**, not three. The broad codec conversion is planned, not already implemented. Other ordered numeric controls need the same deliberate review rather than automatic categorization. Nominal modes retain categorical scores. Every expansion needs a named control-by-control count; the earlier illustrative 2,348-output inventory is not an approved final architecture.

Keep stack style at Normal and oscillator wave-warp mode at None in Stage A. Stack style arranges unison pitches. Wave warping includes sync, bend, pulse width, FM and ring modulation, and is separate from effects-section distortion. UI/display settings and random seeds have no prediction heads.

## Sound-loss training path

Native Vital does not supply derivatives for ordinary backpropagation. The active basic path now estimates local sound-loss gradients from legal native renders. The [bounded qualification](../training/reports/NATIVE_GRADIENT_RESULTS.md) passes: continuous changes improve 24/24 cases and joint changes 23/24. The inverse loss connection is tested, but matched inverse training remains pending. A learned forward helper remains an alternative requiring its own qualification; both current helpers failed.

```text
Predictor training with the qualified native estimator:
  target audio -> predictor -> continuous controls / Gumbel-softmax selections
               -> codec/native render boundary -> real Vital with training MIDI
  target audio ---------------------------------> target spectra
  legal control probes -> real spectral losses -> local gradient estimates
                           measured tensor loss -> backpropagation

Deployment and real-render evaluation:
  audio -> predictor -> rounded numeric controls / argmax categories -> codec -> Vital
```

At the explicit native boundary, serialize hard legal patches and measure their sound losses. Reconnect those measurements to the original raw continuous tensors and straight-through choices using the pinned numerical estimator. Its forward value is the measured native loss; its backward is a local estimate, not an exact plugin derivative. A learned-helper implementation instead keeps its weights frozen and retains derivatives through its inputs. Known MIDI conditions only the evaluator or helper, never the deployed inverse network.

Use **straight-through Gumbel-softmax** for the categorical controls: a hard one-hot choice in the forward pass and a soft approximation in the backward pass. Save its RNG and temperature schedule; use unperturbed deterministic `argmax` for deployment-quality scoring. This supplies an approximate gradient through a category choice, not through native Vital. [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.gumbel_softmax.html)

The helper route has research precedent for audio embeddings. Our performance-conditioned spectral model is a project proposal requiring its own qualification. [Combes et al., Neural Proxies for Sound Synthesizers](https://arxiv.org/abs/2509.07635)

### Objective and measurement contract

The primary objective is multiresolution spectral reconstruction: short windows capture attacks and longer windows capture harmonic structure. Preserve timing, absolute level and left/right information. Add envelope and stereo terms for the observed basic-model errors. Spectral distance is a practical approximation to perceived similarity; listening remains part of acceptance.

```text
training loss = spectral reconstruction
              + weighted envelope and stereo errors
              + optional, limited parameter-supervision anchor
```

Freeze sample rate, FFT sizes, hops, windows, magnitude/log transforms, silence floor, valid-time masks, normalization and weights in a versioned contract. The initial candidate uses the existing 44.1 kHz render path and FFT sizes 512/2048/8192 while retaining time structure. Any frequency/time reduction needed for memory must preserve real-render error ordering and sensitivity to detune, filtering, attack/release, level and pan. The old helper's coarse 24-band/eight-time-bin representation is not a qualified objective by default.

Train the helper against the declared spectral/envelope/stereo measurements. Use its predictions to approximate that objective during inverse training. Report **helper-based training loss separately from loss on real Vital audio**. Low feature regression error alone cannot qualify a helper.

Parameter supervision can initialize the predictor and stabilize early fine-tuning. Declare its scale and decay before comparison and inspect actual gradient contributions. Verify that sound loss alone supplies finite, nonzero gradients to active continuous and categorical heads with the parameter term disabled. Alternative settings that sound equally good must not fail acceptance solely because they differ from the source preset.

## Stages and advancement

### A. Learn basic reconstruction from sound — active

Use one init oscillator, optional filter, amplitude envelope and bounded unison. Retain split manifests, the compact predictor, frozen Basic Pitch features and training-only retrieval. Keep the prediction space fixed while establishing the sound-training mechanism.

#### A0. Contract, gradient correctness and resource profile

The [local contract/native check](../training/reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md) now passes, including 33 focused tests, 12 native requests, six exact repeats and measured target memory. The [local A1 helper card](../training/reports/BASIC_SPECTRAL_HELPER_CARD.md) freezes its implementation, data, qualification gates and resource limits. A0 establishes connectivity and resources, not useful helper gradients.

Implement the spectral targets, differentiable control adapter, Gumbel-softmax selections, legal inactive defaults and separate export path. Check sound-loss gradients, fixed helper weights, hard-choice codec agreement and interrupted-run recovery. Profile real target shapes, replay bytes, optimizer memory and full rendering throughput before sizing a run. Freeze the helper qualification card described in the [basic-stage protocol](../training/STAGED_RECONSTRUCTION.md).

**Deliverable:** reviewed contract, focused correctness checks, measured resource profile and a frozen experiment card. **Exit:** active controls receive sound-loss gradients, hard exports remain valid and retained state fits the budget. These checks do not establish useful gradients or reconstruction quality. Existing `train` commands still use their implemented supervised loss until a separate sound-training mode exists.

#### A1. Qualify useful sound-loss gradients on the small vocabulary

The [first local helper](../training/reports/BASIC_SPECTRAL_HELPER_RESULTS.md) and [aligned-pair correction](../training/reports/PAIRED_SPECTRAL_HELPER_RESULTS.md) have completed. Both fail the useful-gradient gates; neither may guide inverse training. The paired run improves 31/48 guided cases, but saved-data diagnosis finds weak attack/decay/release response fitting even on training pairs. The next work is a targeted response investigation, not another automatic full fit or expanded output space.

The [completed time-coverage investigation](../training/reports/SPECTRAL_TIME_COVERAGE_RESULTS.md) finds that sparse training frames miss some short attack/release changes. Repair and verify temporal supervision before another full fit. This concrete data issue does not establish that the helper architecture or all other controls are adequate; all original native gates remain required.

The [event-frame repair](../training/reports/EVENT_SPECTRAL_FRAMES_RESULTS.md) now passes its native signal-retention and resource checks. Before another full attempt, freeze a short fitting check on real measurements to establish whether the helper can learn the preserved responses. Its architecture and the 14-output inverse scope stay unchanged.

The [completed matched short fit](../training/reports/EVENT_RESPONSE_FIT_RESULTS.md) does not establish learned control responses. Corrected sampling helps absolute error, but attack/decay/release/detune response gates fail even on the inspected training pairs. The next mechanism is the separately costed native-gradient alternative described below, with direct real-audio qualification before inverse integration. Do not spend another full run on this unchanged helper.

Reuse existing collection/recovery infrastructure, not the failed helper's qualification. Train a small temporal forward model using basic patches and known MIDI. Cover register, velocity, note duration and releases within training rather than only the old four fixed performances. Keep qualification families/performances separate. Include nearby legal control changes and hard categorical alternatives in training-only coverage.

Check held-out spectral prediction, rankings of real-rendered candidates, and whether small helper-guided changes improve real Vital audio. Probe continuous and categorical changes separately, then the combined path. Compare train-only mean, nearest-neighbor and interpolation baselines. Audit stereo/detuning sensitivity explicitly.

**Deliverable:** qualified helper artifact and native report. **Exit:** predeclared prediction, ranking and real-improvement gates pass on reference and transfer performances. A helper that improves its own loss while worsening real audio stays disconnected from inverse training.

If qualification fails, diagnose the conflicting measurements/directions and declare one evidence-based change before another bounded attempt. A differentiable implementation of the restricted synth or bounded native optimization is a separately costed alternative, not an automatic successful fallback. Do not automatically enlarge the helper, output space or run budget.

The [native-gradient alternative card](../training/reports/NATIVE_GRADIENT_CARD.md) now freezes that branch: direct finite differences and categorical sound-loss tables, at most 624 native attempts, 30 minutes, 8 GiB RAM and 32 MiB output. It retains the continuous/joint real-improvement gates on reference and transfer, exact repeats and QA. Forward-helper prediction/ranking gates remain required for a learned helper, but do not apply to direct native measurements. Only a passing native direction report permits a separately costed inverse-training trial; implementation alone does not qualify it.

The [completed native result](../training/reports/NATIVE_GRADIENT_RESULTS.md) passes all those gates with 560 accepted renders, 24 exact repeats, 20 minutes and 0.88 GiB RAM. This qualifies the bounded estimator for a costed A2 trial. It does not qualify either learned helper or establish inverse-model quality. Each training example can require 22 native probes, so the next card must freeze a practical fit/update/audit allocation before execution; any revision to the A2 request budget must be explicit.

#### A2. Train the predictor on spectral loss

Start a matched parameter-only control and spectral-loss candidate from the same selected basic checkpoint. Align encoder, output vocabulary, source population, performances and inverse-update budget. Freeze the qualified helper for the whole inverse run, or pin a qualified native-gradient estimator and count every native probe against the card. Use Gumbel-softmax for both basic categorical decisions and differentiable bounded continuous controls.

Infer one patch from a reference recording and compare its native or qualified-helper measurements under reference and additional training performances against real target renders of the same source patch under those performances. Reusing that one inferred patch encourages a playable sound beyond one note. No source patch labels or true MIDI enter the inverse network.

Run a small end-to-end fit check because the objective changed, then the bounded matched comparison. Render hard-decoded predictions through Vital at predeclared checkpoints. Monitor category usage, sound/parameter gradient contributions, helper/native disagreement and invalid outputs. Any helper refresh needs a new qualified artifact and comparison, not simultaneous untracked changes to both models.

**Deliverable:** spectral-trained candidate, matched control, recovery evidence and native report. **Exit:** real reference/transfer improvement over the matched parameter-only control and simple baselines within budget. Better helper loss cannot advance the stage. Report helper training as additional compute; matched inverse updates do not imply equal total compute.

#### A3. Confirm basic quality and package the first bounded model

Freeze the candidate and comparison rules, then use new independent validation families and performances. Preserve all old assignments and test reservations. Confirm hard-export performance, tail errors, transfer behavior and blinded listening. Replicate a promising learning change on a second seed; do not run seed sweeps on an unqualified mechanism.

Prepare a basic inference bundle requiring neither the helper nor source MIDI at prediction time. Keep a real Vital render as the end-to-end sound check. Listening, recovery and fresh-validation gates must pass before basic reconstruction is complete.

Exact categorical recovery may remain a tiny-set implementation check; it is not a perceptual score or a requirement to recover non-identifiable graphs and seeds.

### B. Add a bounded sound capability

After A passes, choose the first missing capability from its failure analysis: for example a small factory waveform vocabulary, filter-envelope control, or a second oscillator, then a third if useful. Do not activate all of them together. Freeze a new source population, named output inventory and model-size budget before training. Requalify the helper on every expanded vocabulary before using its gradients there.

Compare the previous-stage model and the expanded model on the same new sounds, and retain an earlier-stage panel to detect regressions. Score the original audio; separately measure the expanded representation's ceiling. The restricted model's inability to express a sound is a measured baseline, not grounds to exclude that sound from the comparison.

### C. Add simple modulation

Introduce one LFO and at most two linear source-to-destination connections. Begin with a single source and a few audibly important destinations such as oscillator level, pitch, pan and filter cutoff. Source and destination vocabularies must be declared explicitly, not inherit all 431 destinations. Use observation lengths that expose at least the relevant modulation cycle and evaluate under changed tempo, note length and retrigger conditions.

For this restricted, independent graph, authored slot order can provide convenient supervised targets. Audio under multiple performances decides quality. Two different valid graphs that reconstruct equally well must not be ranked by exact source graph recovery. Do not apply unrestricted graph permutation matching to the comprehensive codec, where indexed references and order can matter.

Only if simple shapes leave a demonstrated gap should this stage compare one train-fitted complex LFO head. Eight simultaneous complex curves are not the starting point. Requalify the sound-training path and preserve earlier simple cases throughout training and evaluation. Keep mappings linear; the optional first-two-route/six-knot trial above remains a late experiment.

### D. Expand selectively and prepare a useful release

Add effects, sampler, additional oscillators or further modulation one capability at a time, selected by observed usefulness and cost. Each addition needs the same controlled quality comparison. Route maps and custom oscillator curves remain optional late experiments. Exhaustive Vital configuration recovery is not a milestone.

A release can support a deliberately bounded sound vocabulary. Its model card must state that scope and how out-of-scope inputs behave. Freeze weights, preprocessing, assets and inference policy, then run the untouched final test. If test results drive further changes, treat them as development evidence and reserve new confirmation data rather than repeatedly calling the same panel an untouched test.

Package trained weights, preprocessing, the exact stage decoder/defaults, required asset hashes/notices, an offline inference CLI, runtime measurements, reproducible local training instructions and listening/evaluation results. Verify a fresh local-environment example. Colab packaging and native Linux qualification are deferred and are not local-release prerequisites. A GUI is optional. The deployed inverse model does not need the training helper.

## Sound-quality decisions

Infer a patch once from a reference recording. Compare its real Vital audio with the source under both the reference performance and reserved transfer performances. Ground-truth MIDI belongs to the training forward model and synthetic renderer/evaluator; it is never an inverse-model input. All baselines use the same cases and performances.

Required comparisons are the fixed template, training-only retrieval, and the unchanged previous model. Primary reporting uses per-patch reference/transfer spectral error; report reference and transfer separately. Add amplitude-envelope and level error, stereo diagnostics, median and tail errors, and all failed/invalid/silent/clipped outputs. A lower spectral metric is evidence, not a calibrated claim of perceptual similarity.

Use equal weights per independent patch family and paired family-level intervals. Do not count correlated performances or variants as independent successes. Predeclare candidates, selection rules, failure handling, render budgets and stopping rules before running a comparison. Failed cases stay in the denominator and block a clean promotion; they are not silently dropped from the mean.

Stage completion requires:

- Valid loadable patches, finite training, frozen pretrained weights/buffers, correct splits and verified recovery.
- Better held-out reconstruction than template and retrieval; an added capability must also improve on its prior model. Freeze a practical margin before comparison (initial working target: 10% lower primary error), with a paired interval excluding no benefit on the confirmation panel. Report any missed target directly.
- Reference and transfer results both acceptable, with no hidden regression in earlier sound families. Predeclare a tail/earlier-stage regression tolerance after inspecting metric calibration, before seeing the candidate comparison.
- Blind listening evidence supporting similarity and usefulness. If objective gains do not sound better, diagnose the metric or model rather than declare the stage complete. Prepare randomized clips and record the actual listener results; automated scores cannot substitute for this step.
- Measured memory, throughput and inference cost within the local budget, or an explicit unresolved limitation.

These are decision rules, not established universal perceptual thresholds. If intervals are too wide, obtain more independent confirmation cases before multiplying training seeds. Replicate a promising structural change on a second seed before promoting it; do not perform repeated seed studies while the baseline failure remains unexplained.

Final-test states remain untouched until the release candidate and all choices are frozen. The final test is for the declared supported vocabulary; it does not need every parameter the preset codec can encode.

## Data, rendering and reproducibility

Audio is generated on demand, measured and discarded in the training pipeline. Keep bounded feature replay, recipes, QA logs and recoverable checkpoints; a pre-rendered training WAV collection is not required. Small listening/evaluation clips are allowed. Existing inverse-feature caches may not contain the new spectral targets: reuse only with matching complete contracts, otherwise regenerate on demand under a new identity. Render source presets faithfully for the stage and keep ancestry-aware splits immutable.

Start with one isolated native worker and bounded prefetch. Measure the full path (plugin startup, state loading, offline rendering, features, updates and checkpoint writes), not just synthesis time. Rendering can run faster than real time; optimize the measured bottleneck. Increase workers only after resource and state-isolation checks.

For exact reproducibility diagnostics, set oscillator 1–3 random phase and sampler random phase to fixed values, preserve all other phase controls and seeds, and use a fresh process per request. Log controlled copies separately. The first new spectral-training experiment also declares fixed randomness for source and predicted renders. Version this training transformation, preserve original lineage/splits, and rerender baseline and candidate under the same policy. Do not relabel historical random-phase features or scores as controlled data.

Checkpoints must include helper identity, measurement/loss contract, tensor adapter, optimizer/scheduler, minibatch and Gumbel RNG, temperature schedule, replay and render cursor. Verify exact recovery on the same qualified runtime using retained tensors. CPU/GPU or cross-platform transfers require their own numerical checks; bit-identical cross-device recovery is not promised.

When randomness is intentionally introduced in a later stage, assess sound statistics or a declared repeated-render score. Do not add 96 binary seed heads to make waveform identity an optimization target.

## Free-first compute and experiment limits

Run the upcoming implementation, qualification and training stages entirely on the local machine. Colab work is deferred at the user's request. No paid GPU, subscription or storage is assumed. Start with CPU float32, two learning threads, one renderer and an **8 GiB process-tree RAM budget**. Keep the existing 355,806-parameter inverse model. The initial helper ceiling is two million trainable parameters, a planning limit to profile rather than a claim of sufficient capacity.

Replay is capped by **both bytes and records**: at most 512 records and 512 MiB resident features/targets, whichever fills first; optional retained feature archives are capped at 2 GiB. Full multiresolution spectra are much larger than old pooled inputs. Profile batches of 1–8 examples and reduce batch/replay size or use declared time chunks before exceeding memory. Any lossy target compression needs a fidelity check. Account for feature extraction, checkpoints, helper training and evaluation separately.

These initial ceilings are caps, not runtime predictions or a launch instruction. A frozen experiment card is required before each run. Failed requests and recovery attempts count inside the caps. Stop at the first reached limit and report incomplete coverage. A revised budget requires a new declared experiment, never an automatic extension after inspecting results.

| Work package | Native requests, including audits | Optimizer updates | Wall time |
|---|---:|---:|---:|
| A0 correctness and resource profile | 96 | 500 debugging updates | 30 minutes |
| A1 helper feasibility | 2,048 training + 256 qualification | 12,000 helper updates | 4 hours |
| A2 matched inverse comparison | 1,024 total | 8,000 per arm; 16,000 total | 4 hours |
| A3 confirmation evaluation | 960 | None during evaluation | 2 hours |

Second-seed training is a separate predeclared run capped like A2, after a promising result. New stages need their own cards under the same resource policy. Choose confirmation sample size from development variability before scoring the candidate; if adequate coverage cannot fit the cap, revise the plan before collecting confirmation.

The recent basic diagnosis rendered 290.73 seconds of audio in 105.28 seconds end to end; native synthesis alone took 55.53 seconds. This demonstrates faster-than-real-time operation for that workload, not promised training throughput. The earlier helper run took 18.85 minutes, mostly rendering. Profile the complete new path before increasing workers or moving learning to a GPU.

Colab remains an optional later scaling route, after local basic reconstruction works and a measured bottleneck warrants it. Do not spend the current stages on notebooks, cloud transfer or Linux plugin qualification. If revisited, account for non-guaranteed hardware/session availability and qualify portability then. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Keep GitHub Actions disabled and run relevant checks locally. Submit reviewable PRs for A0, A1, A2 and confirmation/release work; never merge without direct approval. This plan update does not launch experiments or promote an existing helper or predictor.
