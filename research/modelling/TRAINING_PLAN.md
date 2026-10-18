# OBRUXO staged reconstruction plan

Revised 2026-10-19 following the scope correction. **Prove sound reconstruction with a small prediction space, then add capabilities when they improve audible results.** Comprehensive preset support is useful infrastructure, not a requirement to predict every field. This plan supersedes the [coverage-first snapshot](TRAINING_PLAN_COVERAGE_20261018.md), including its full-model fitting prerequisite and exhaustive coverage milestones.

The [completion ledger](../training/IMPLEMENTATION_STATUS.md) tracks the active work. The [basic-stage protocol](../training/STAGED_RECONSTRUCTION.md) fixes the immediate scope and next decisions. Prior code, checkpoints, reports and split reservations remain available.

## Target and current evidence

One isolated instrument recording enters; one valid, playable `.vital` patch leaves. Inference receives audio only. The useful result is a similar sound, including its response to new notes, durations and velocities. Recovering the exact source settings or routing graph is not the product goal.

The small problem has partial evidence already: the nine-control/two-choice pilot learned from 205 and later 404 training patches. Its [original validation](../training/reports/STATE_COVERAGE_RESULTS.md) and [additional validation](../training/reports/FRESH_VALIDATION_RESULTS.md) show reconstruction gains, but persistent envelope, level and unison errors. Neither proves reliably good reconstruction or listener usefulness. Both panels (21 and 23 patches) have been used for development. They must not be relabelled as new confirmation data.

The latest eight-example full-model run is an infrastructure diagnostic, not the next training prerequisite. Further work on fitting all outputs, expanding route coverage or balancing eight codebook losses is deferred. The next job is to diagnose the small model's sound errors, establish a useful baseline and close that stage's quality gate.

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

## Stages and advancement

### A. Reliable basic reconstruction — active

Use one init oscillator, optional filter, amplitude envelope and bounded unison. Retain the existing split manifests, compact model, frozen Basic Pitch features and training-only retrieval baseline. Keep the architecture and prediction space fixed while diagnosing its present limits.

1. Audit existing results and choose representative good, median and poor cases before new training. Separate amplitude/envelope, spectral shape, stereo/unison and performance-transfer errors. Reuse existing checkpoints and feature caches when identities match.
2. Confirm the small end-to-end fit and native serialization evidence. Re-run a roughly 32-example check only if the training path, features, objective or representation changes. Render its predictions; low parameter loss alone does not pass a sound-quality gate.
3. Make one substantial, diagnosis-driven change to observation, data coverage or sound-based learning. Compare it with the unchanged basic model on the same training budget and development cases. Do not launch another seed or loss-weight sweep without a concrete unresolved question.
4. Freeze the candidate and comparison rules, then assess new validation families and performances that were not used to choose the change. Preserve every existing test reservation. Prepare a blinded listening comparison as part of stage completion.

Parameter supervision is a bootstrap and debugging signal. Exact categorical recovery may remain a tiny-set implementation check for these authored, restricted controls; it is not a perceptual score or a prerequisite to recover non-identifiable graphs and seeds.

### B. Add a bounded sound capability

After A passes, choose the first missing capability from its failure analysis: for example a small factory waveform vocabulary, filter-envelope control, or a second oscillator. Do not activate all of them together. Freeze a new source population, stage-specific output vocabulary and model-size budget before training.

Compare the previous-stage model and the expanded model on the same new sounds, and retain an earlier-stage panel to detect regressions. Score the original audio; separately measure the expanded representation's ceiling. The restricted model's inability to express a sound is a measured baseline, not grounds to exclude that sound from the comparison.

### C. Add simple modulation

Introduce one LFO and at most two linear source-to-destination connections. Begin with a single source and a few audibly important destinations such as oscillator level, pitch, pan and filter cutoff. Source and destination vocabularies must be declared explicitly, not inherit all 431 destinations. Use observation lengths that expose at least the relevant modulation cycle and evaluate under changed tempo, note length and retrigger conditions.

For this restricted, independent graph, authored slot order can provide convenient supervised targets. Audio under multiple performances decides quality. Two different valid graphs that reconstruct equally well must not be ranked by exact source graph recovery. Do not apply unrestricted graph permutation matching to the comprehensive codec, where indexed references and order can matter.

Only if simple shapes leave a demonstrated gap should this stage compare one train-fitted complex LFO head. Eight simultaneous complex curves are not the starting point. Preserve earlier simple cases throughout training and evaluation.

### D. Expand selectively and prepare a useful release

Add effects, sampler, additional oscillators or further modulation one capability at a time, selected by observed usefulness and cost. Each addition needs the same controlled quality comparison. Route maps and custom oscillator curves remain optional late experiments. Exhaustive Vital configuration recovery is not a milestone.

A release can support a deliberately bounded sound vocabulary. Its model card must state that scope and how out-of-scope inputs behave. Package trained weights, preprocessing, the exact stage decoder/defaults, required asset hashes/notices, an offline inference CLI, runtime measurements, reproducible training instructions and listening/evaluation results. A GUI is optional.

## Sound-quality decisions

Infer a patch once from a reference recording. Compare its real Vital audio with the source under both the reference performance and reserved transfer performances. Ground-truth MIDI belongs only to the synthetic renderer/evaluator; it is never an inference input. All baselines use the same cases and performances.

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

Audio is generated on demand in the training pipeline. Keep bounded feature replay, recipes, QA logs and recoverable checkpoints; a pre-rendered training WAV collection is not required. Reuse features only with verified audio, preprocessing and recipe identities. Render source presets faithfully for the stage and keep ancestry-aware splits immutable.

Start with one isolated native worker and bounded prefetch. Measure the full path (plugin startup, state loading, offline rendering, features, updates and checkpoint writes), not just synthesis time. Rendering can run faster than real time; optimize the measured bottleneck. Increase workers only after resource and state-isolation checks.

For exact reproducibility diagnostics, set oscillator 1–3 random phase and sampler random phase to fixed values, preserve all other phase controls and seeds, and use a fresh process per request. Log the controlled diagnostic patch separately. Ordinary training/evaluation uses its declared stage phase/stochastic policy; diagnostic phase fixing must not silently alter the task or its targets. Exact checkpoint recovery uses retained feature tensors, not a claim that every native rerender is bit-identical.

When randomness is intentionally introduced in a later stage, assess sound statistics or a declared repeated-render score. Do not add 96 binary seed heads to make waveform identity an optimization target.

## Free-first compute and experiment limits

The default is the local machine with free Colab as a fallback. No paid GPU, subscription or storage is assumed. The current local baseline uses CPU float32; the working process-tree budget is 8 GiB. The small model's historical 205-state run took about 8.8 CPU minutes including collection, and the 404-state feature replay used about 14.9 MiB. These are prior measurements, not estimates for every new run.

Use two CPU training threads initially and one renderer; account for feature extraction and evaluation separately. Keep the current 512-record replay cap unless measured data requirements justify changing it. Stop a scheduled experiment at its frozen update, fresh-render and wall-time limits and report incomplete coverage. No automatic budget extension after inspecting results.

Free Colab is optional, with CPU rendering and GPU learning only after Linux Vital loading, native output behavior, restart/checkpoint recovery and full-path throughput are qualified. GPU availability is not assumed. If rendering remains local, transferring verified feature batches is a separate recorded workflow with pinned provenance. Do not present unqualified Linux rendering as a working portable training path.

Keep GitHub Actions disabled and run relevant checks locally. Submit reviewable PRs; never merge without direct approval. The next PRs should close basic reconstruction gaps, not add exhaustive output coverage.
