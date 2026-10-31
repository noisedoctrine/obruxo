# Test identical musical coverage across the three sounds

This is the next A2 fit after the [failed auxiliary comparison](SHARED_BASIC_AUXILIARY_RESULTS.md). Change the primary training recordings while retaining the same three patches and 14 outputs. The [balanced source bank](BALANCED_BASIC_DATA_CARD.md) must pass its native/feature audit first. Do not add an adversarial loss or expand the vocabulary.

## Fixed comparison

Reuse the completed, independently verified **sound-only** arm as the control; pin its diagnostic export, initializer, final comparisons and verification hashes. Start the candidate from the same original **untrained** trainable-input bridge, never from either rejected fitted model. Keep the same trainable mask, frozen baseline/frontend, inward control derivative, Adam at 0.001, norm cap five, production argmax, backward temperature schedule, native estimator and spectral/envelope/stereo objective.

Keep **48 updates** and exactly the preceding control's patch order and two reference selections at every update. Each selected patch therefore receives sixteen primary updates. Input one remains its original recording. Input two cycles through its six balanced recordings in this fixed order of grid indices: **0, 3, 4, 1, 2, 5**, repeated as needed separately for each patch. Grid indexing is pitch-major, then short/long duration. All three patches receive identical counts: eight short and eight long grid inputs, with low/middle/high pitch counts 6/5/5. Every grid cell appears two or three times. This is identical musical coverage between patches; 16 draws cannot be perfectly uniform over six cells.

For both primary inputs, render its predicted patch under the original and currently selected grid performances, using the same reciprocal two-playback objective as the control. Keep primary weights 0.25/0.5 and two round-robin reference weights 0.125 each, including the existing accepted-initial reference hinge. The candidate receives sound loss only; no source-control labels or auxiliary note loss enter its optimizer.

Keep the existing actual-sound acceptance rule: a 0.025 continuous-control cap, at most three measured trials, improvement greater than 1e-6, exact model rollback on refusal and one Adam advance per scheduled update. Apply the unchanged cap rule over the input bank including the new recordings: 86 unique inputs, versus the control's 68. This added coverage is explicit, rather than silently omitting newly trained inputs from the safety bound. Native gradients are fresh at each changed state.

## Evaluation and decision

Retain the same old **104 comparison cells**, including all 21 selected-patch comparisons, six full-panel groups and the original 32 reference patches. Reuse their exact recorded initial/control endpoints. Also evaluate each of the eighteen grid inputs under its own and original performance: 36 new comparison cells for the initializer, stored control and candidate. Collect grid initializer endpoints before training; collect fitted-model endpoints only after training. Grid results are training/development evidence, not fresh validation.

The three-case prerequisite remains unchanged: each patch's mean combined loss improves at least 25% from initialization, every existing comparison improves combined loss by at least 10% and MRSTFT strictly, and both held-input playback losses are no worse than the saved settings-trained control. Keep the existing six-group criteria, minimum 36 accepted updates, active-head/input gradient checks and stronger earlier-32 mean/tail preservation. Do not relax these gates to admit a result.

Report directional data benefit separately: at least 5% lower combined loss and MRSTFT on the old 21 cells versus the audited sound-only control, no individual patch's mean metric more than 5% worse, and the existing earlier-32 preservation gate. Report each input and patch even if the aggregate improves. On the new grid, require each patch's mean combined loss to improve at least 10% from initialization, beat the control mean and lower MRSTFT. These are additional fit checks; they cannot substitute for the old gates. Only the full unchanged admission requirements can advance A2.

## Resources and verification

Use the same qualified Vital identity, fixed phases for all oscillators and the sampler, 32,768-frame preroll, CPU float32, two Torch threads and one isolated native worker. The source bank's 24 requests remain separately recorded. Candidate allocation:

| Work | Maximum requests |
|---|---:|
| 48 updates × six playback observations × 22 gradient queries | 6,336 |
| 48 updates × six observations × three trial scales | 864 |
| Old candidate endpoints and six held-input exact repeats | 110 |
| 36 grid cells × initializer/control/candidate | 108 |
| Planned total | **7,418** |
| Hard cap including recovery | **8,000** |

Allow six hours, 4 GiB process-tree RAM, 2 GiB evidence and 512 MiB resident replay. Cap training requests at 7,600 and qualification at 400, two native executions and 49 charged update attempts. Permit only one known identical timeout recovery under the original deadline. Never silently restart or extend the run. Checkpoint candidate, Adam, trace and RNG after every update; exactly replay the first two native updates before continuing.

The offline audit has twenty minutes / 4 GiB and zero native requests or experimental fitting. Reconstruct every update and optimizer/RNG state, cold-load the candidate, validate all receipts, rescore all **426 comparison rows** (old initial/control/candidate, new grid initial/control/candidate, and six candidate repeats), and require exact fixed-phase repeats. Training probe audio remains transient; describe scalar replay separately from retained waveform rescoring.

Freeze the complete machine-readable plan and source/input hashes before execution. Local checks, real-initializer recovery and unchanged old starting predictions must pass before launch. Publish failed gates as well as gains. The 192 reserved roots, final test, second seed, clean human listening and model release remain conditional on A2; Actions stays disabled and Colab remains deferred.
