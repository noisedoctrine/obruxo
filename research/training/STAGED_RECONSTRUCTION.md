# Basic reconstruction: active protocol

Declared 2026-10-19. This implements the immediate scope of the [revised plan](../modelling/TRAINING_PLAN.md). The broad preset infrastructure remains available. No broad training or curve-loss tuning is scheduled by this protocol.

## Prediction contract

Use the existing `PilotCodec`, including its bounds, fixed template, activity masks and identities. Predict oscillator level/pan/unison detune, filter cutoff/resonance, envelope attack/decay/sustain/release, unison voices (1/2/4) and filter on/off. This is nine continuous outputs and categorical sizes three and two: **14 output values, 355,806 trainable parameters**.

The basic model has no learned assets, LFO curves, route slots, mapping curves, custom oscillator curves or seed bits. Other source controls retain the existing template values. The init asset and default patch are part of the saved codec contract. No parameter family is silently removed from complex source audio.

`train --prediction-scope basic` checks the exact basic codec before initializing features, training or rendering. Supplying a broad pool fails. The scope is saved in the training contract; changing it on resume fails. Legacy checkpoints retain their original contract under the default `codec` option. Verified same-pool feature seeding can start a new explicitly basic run with fresh learning state; it does not relabel a historical checkpoint.

For a new installation, the existing commands can reproduce the small plumbing check:

```powershell
python research/training/run.py pilot --count 256 --seed 0 --output research/training/outputs/basic-pool
python research/training/run.py train --prediction-scope basic --pool research/training/outputs/basic-pool/pool.json --output research/training/outputs/basic-fit --steps 800 --replay-capacity 32 --batch-size 16 --render-budget 64 --checkpoint-every 100 --cpu-threads 2
```

This is a reproducibility recipe, not an instruction to repeat a completed experiment. Locally, reuse the existing pools/checkpoints below. The historical tiny-fit gate remains an implementation diagnostic; stage advancement uses rendered sound and listening evidence.

## Existing baseline and immediate diagnosis

Retain the selected seed-0 checkpoint at `outputs/state-coverage-expanded-16000-20261013/checkpoint.pt` and its 404-training-state pool `outputs/authored-expanded-20261013/pool.json`. Its inference bundle remains `outputs/state-coverage-validation-20261013/selected-model.pt`. Use the original 205-state retrieval checkpoint pinned in the historical reports. Do not change the incumbent by selecting again across old checkpoints.

The [evidence audit](reports/BASIC_RECONSTRUCTION_AUDIT.md) reuses the published 21- and 23-patch development results. Its machine-readable report records input hashes and selects five diagnostic cases at the minimum, lower quartile, median, upper quartile and maximum of the additional panel's existing sound error, with deterministic tie handling. This selection is fixed before any new intervention render. It samples a range of errors, not just the worst examples.

Next native diagnostic, before another training run:

- For each selected case, retrieve the original patch and saved baseline prediction by the recorded identities. Verify the historical checkpoint, case and prediction receipts.
- Under each recorded reference/transfer performance, render source, source repeat, baseline prediction, and four separate diagnostic corrections. Each correction replaces one control group with its source values: amplitude envelope; oscillator level; filter enable/cutoff/resonance; or unison voices/detune/pan. Canonicalize inactive values and validate the resulting patch.
- Fix all three oscillator random-phase controls and sampler random phase for every diagnostic render. Preserve other phase controls and seeds. Record these controlled copies separately; use a fresh native process for each request. This is an explicitly controlled diagnosis, not a replacement of historical task scores.
- Score each correction against the same controlled source and baseline. Compare spectral and level/envelope/stereo errors, retain all failures and prepare representative listening clips. A source-value intervention uses oracle information and is never a deployable prediction or a new model-quality result.

The bounded diagnostic is **5 cases × 2 performances × 7 methods = 70 requests**, zero optimizer updates, one worker, at most 8 GiB process-tree memory and a 15-minute wall-time cap. Record a stopped/incomplete run rather than silently increasing the budget. No retries are automatic. An explicit recovery can preserve completed requests and log any replacement attempt. Exact repeats diagnose pipeline correctness; an error is investigated rather than averaged away.

These group interventions locate useful directions but can interact and are not an additive decomposition. If no group helps, inspect observations, representation limits and the metric instead of assuming a parameter loss needs more weight. A large parameter error alone is insufficient evidence to choose a training change.

## Next learning comparison and confirmation

Choose one substantial correction from the native diagnosis. Before running it, write a separate experiment card that fixes the hypothesis, source/split/performance identities, baseline checkpoint/retrieval, changed code, update/render/wall-time caps, quality margin and failure rules. The unchanged basic prediction space is the default. Do not start a new architecture, loss-weight or seed sweep under this diagnostic protocol.

Both existing panels are development data. For confirmation, append independent authored families with the same basic vocabulary and immutable split inheritance; use only newly assigned validation families. Verify against all existing training/development identities and reserve every test assignment. Freeze the complete validation membership and performances before candidate scoring. The confirmation population size and render budget must be fixed from the observed development variance, not adjusted after seeing a favorable candidate score.

Use template, training-only retrieval and unchanged model controls. Report reference/transfer error separately, failure rates, median/tail behavior and uncertainty by independent family. The working practical target is a 10% primary-error improvement, with a paired interval excluding no benefit and declared earlier-stage/tail regression limits. No claim of stage completion follows solely from parameter accuracy or a lower training loss. A promising change then gets a second-seed confirmation and blinded listening assessment.

Only after basic-stage quality passes should the next PR introduce a bounded new capability. One LFO and at most two linear routes are the initial modulation limit. Optional compact mappings for the first two routes remain deferred; no 64-slot mapping output layer is part of this schedule.
