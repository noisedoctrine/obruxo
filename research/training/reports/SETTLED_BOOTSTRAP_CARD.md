# Corrected-render basic initialization: small fit check

Frozen before execution, 2026-10-22. The [paired input comparison](RENDERER_INPUT_COMPARISON_RESULTS.md) found a substantial change in the historical initial model's inputs after correcting Vital's startup. Refresh initialization on corrected audio, beginning with a small training-only check. This does not expand the 14 outputs or replace perceptual training and the unchanged held-out quality gates.

## Fixed scope

Use the existing authored 404-training-state pool, its original split reservations and the first eight distinct training states in the seed-zero `state_round_robin` schedule. Preserve their source and performance identities. The adapter fixes oscillator 1–3 and sampler random phase to zero and uses the qualified 32,768-frame silent lead; MIDI is shifted exactly and only the added lead is trimmed. Source identities continue to describe the original patch; the recipe and request identities explicitly include the controlled render setup.

The model remains nine continuous controls plus five category scores, 355,806 trainable parameters. Start a fresh seed-zero model with the same frozen frontend. Collect eight reference inputs through the training loop; retain only features, labels and receipts, with no WAV or spectral-target archive. Cold-render features, checkpoints and recipe cursors cannot seed this run.

Train on parameter labels solely to check initialization: 800 AdamW updates, learning rate 0.001, batch size eight, float32 CPU, two threads. Use final weights only. Keep a checkpoint after every collected example and every 50 optimizer updates, plus the previous atomic checkpoint. New CLI basic training defaults to corrected rendering; explicit zero lead preserves historical reproduction.

## Native check and limits

Audit all eight training states using the same one predicted patch per source. For each state, render reference and reserved transfer performances for the source, literal source repeat, model, fixed template and training-only retrieval: ten requests per state. Transfer notes are a diagnostic on these same training patches, not fresh validation.

Planned cost: eight training renders plus 80 audit renders = **88 native requests**. Original-run limits: 30 minutes wall time, 4 GiB process-tree RAM, 256 MiB archive, 800 committed optimizer updates. Explicit recovery may consume at most two additional source attempts and ten additional audit attempts, for **100 total attempts**. Charge each native request durably before renderer startup. There are no automatic retries or budget extensions. Resume preserves the original deadline, uses a new training segment and retains all prior attempts and outcomes. An attempt without a corresponding outcome prevents a clean pass.

One fresh native worker per request, with 60-second request timeout. Check wall/RAM limits at each training update and render boundary, and archive size at each progress boundary; these are cooperative limits with at most the in-flight request between checks. Resource measurements must be available. The user has authorized running on battery with checkpoints; do not alter host power settings.

## Admission

All must pass:

- The tiny-fit check: active continuous MAE below 0.05, categorical accuracy 1.0 and final label loss below 5% of its initial value.
- Every charged attempt has a matching accepted outcome, and the renderer identity matches the qualified renderer.
- Every source repeat has zero native MRSTFT distance on both performances.
- Mean model MRSTFT is below 90% of the fixed-template error on both performances.
- Frozen source membership, render profile and resource limits hold.

Publish scores and failures without extending this run. A pass permits freezing a larger corrected-render initialization run; it is not evidence of held-out sound reconstruction or a release. The next sound-loss experiment must recheck category calibration against the new initial weights. A failure requires a specific diagnosis before more fitting.

## Commands

Use `reports/scripts/run_basic_bootstrap.py` with `--pool`, `--qualification` and `--output` to save the hash-bound plan without rendering. Add `--execute` after reviewing that plan. Use `--execute --resume` only for an interrupted run. The public plan, source commit, local checks and eventual result accompany the ready PR; private model/cache artifacts stay in the local output directory.
