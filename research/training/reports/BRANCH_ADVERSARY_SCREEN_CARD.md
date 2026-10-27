# Cheap directional test: remove note information from the timbre encoder

The [original staged plan](../../archive/agent_plans/next_steps.md#phase-1-disentanglement-target-after-phase-0-is-stable) calls for learning the same patch across different performances. The [encoder proposal](../../modelling/pretrained%20encoders/pretrained_encoder_research.md) also proposes a temporary note predictor whose gradient is reversed at the timbre branch. The [completed probes](BASIC_REPRESENTATION_PROBES_RESULTS.md) now show why this deserves a focused test. This card is fixed before training the new arms.

This is an **encoder-only directional experiment**. It tests whether explicit note removal reduces leakage while retaining patch information. It does not train the inverse model on sound, evaluate reconstructed audio or advance A2. Spectral reconstruction remains the primary objective for the subsequent inverse-model experiment. A successful result here permits preparing that experiment, not promoting a model.

## Fixed comparison

Start from the same qualified untrained basic initializer. Copy only its small timbre CNN into each arm; preserve the original model, frozen Basic Pitch frontend, performance branch, decoder and fourteen-output interface. Both arms train the timbre branch to associate two different performances of the same patch, with other patches in the minibatch as negatives. This branch-specific comparison differs from the older contrastive loss on the combined decoder representation.

Both arms also fit an identical temporary two-output head to the known mean pitch and mean note length. In the control, its gradient stops before the encoder. In the intervention, the encoder receives the opposite gradient, discouraging that information. The head itself always learns the normal prediction task. This uses the [gradient-reversal mechanism](https://jmlr.org/papers/v17/15-239.html); whether it helps this synthesizer task is the hypothesis. Related attribute-removal work includes [Fader Networks](https://arxiv.org/abs/1706.00409), which does not establish an audio result for this project.

Use normalized targets `(mean MIDI - 36) / 48` and `mean note seconds / 3`, squared error, a 768→32→2 GELU head and per-example layer normalization with no learned statistics. Exclude velocity: the current template ignores it. These **two temporary training outputs never enter the preset codec or inference artifact**.

Use all 1,212 distinct cached recordings from the original 404 training patches. Keep every patch's three performances together. Exclude the twelve controlled-grid patches from encoder training, then split the remaining states deterministically within source categories: two thirds for encoder/probe fitting, the rest for probe evaluation. All twelve controlled patches remain excluded. They and the other held-out patches were previously seen during the original baseline initialization; this is not a fresh final-validation claim.

Both arms use the same initial encoder/head/Adam states and exactly the same minibatches. Warm the heads for 100 updates on detached, unchanged encoder features. Then run **500 encoder updates per arm**, 32 distinct patch pairs per update, contrastive temperature 0.1, encoder Adam learning rate 0.0001, head Adam learning rate 0.001 and gradient-norm cap 5. The intervention's reversal coefficient rises linearly from zero to one over the first 100 encoder updates; the control stays at zero. Use seed zero, CPU float32 and two threads. Do not change these settings based on results or select an earlier checkpoint.

## Evaluation and decision

Refit independent probes after encoder training; do not judge leakage solely by defeating the training head. Use the fixed ridge readout and a newly initialized 768→64→2 GELU predictor trained for 250 full-batch updates at 0.001. Fit standardization and all probe weights on training patches only. Evaluate pitch and duration on excluded patches, alongside five within-patch shuffled-label ridge controls.

Also retain source-category readouts, all nine source-active continuous-control readouts, cross-performance patch retrieval and representation spread. On the already-rendered controlled grid, report separate pitch/duration factors and cross-pitch patch identity, using both original and constant-gain LUFS-matched features. Velocity accuracy remains uninformative and cannot count as success.

The intervention is promising for a sound trial only if **all** these predeclared cached holdout conditions pass against the paired-only control:

- Both fresh probes have at least 10% higher pitch MSE and 10% higher duration MSE, indicating reduced readability.
- Each source-category balanced accuracy falls by no more than five percentage points.
- Mean active continuous-control readout MSE rises by no more than 10%.
- Mean cross-performance patch retrieval falls by no more than five percentage points.
- No zero vectors appear and normalized representation spread remains at least half the control's.

These are directional screen rules, not proof of absence of information or perceptual reconstruction quality. Retain the untouched encoder as an additional reference. A failed setting does not reject the general separation objective; report failures without launching a weight sweep automatically. The twelve-patch grid is descriptive supporting evidence, not a replacement for the holdout criteria.

## Compute, recovery and evidence

The combined allocation is **twenty minutes / 4 GiB process-tree RAM / 64 MiB evidence**, with **zero native requests**. Archive the protocol, input/source hashes, state split, exact sampled-index digests, losses and encoder/head/Adam checkpoints. Save after every 100 updates. At update 251, verify the next minibatch and exact encoder/head/Adam recovery from the saved update-250 checkpoint; this adds one software replay update per arm. Fresh-probe optimization is reported separately from encoder and inverse-model training.

Research encoder exports carry their own refused-for-inverse-inference format. The original initializer, input cache and controlled recordings remain unchanged. No source controls or MIDI enter the deployed predictor. No final-test or reserved 192-family confirmation data is used. Keep local checks and GitHub Actions disabled. The real sound-loss fit, complete reconstruction gates, clean listening and final bundle remain unfinished.
