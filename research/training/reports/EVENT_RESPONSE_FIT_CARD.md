# Short matched response fit before another full helper run

Declared 2026-10-20 before fitting. The [event-frame preflight](EVENT_SPECTRAL_FRAMES_RESULTS.md) establishes that the known attack/release changes remain visible in the repaired measurements. This check asks whether the current helper can learn those responses on a small training set. It uses no native renders, no validation/test examples and no inverse-model updates.

Reuse all 12 preflight pairs / 24 recordings, with immutable recipe, ancestry, MIDI and feature hashes. Compare two fresh helpers: the original uniformly sampled measurements and the repaired event-frame/full-envelope measurements. Both use the same 1,413,033-parameter architecture, seed-zero initialization, pair order, seeded minibatch selections, Adam 0.001, four pairs per update, gradient-norm cap five and **1,000 update attempts per arm**. The objective remains absolute spectral/envelope loss plus 0.25 × signed paired-change loss. Save every 100 committed updates and evaluate only the final checkpoint. No sweep or checkpoint selection.

The fitting loop is shared with the full helper screen. Full screens keep their original 90% complete-pair admission floor; this explicitly named small diagnostic does not relabel 12 pairs as a full training population. Both helpers are scored against the **same repaired training measurements**, including complete envelopes. These are fitting results on known training examples, not held-out reconstruction or helper qualification.

To permit consideration of another full fit, the repaired arm must:

- Beat the no-sound-change baseline by at least 20% in mean signed-response error separately for attack, decay and release, and beat that baseline on at least two of three pairs in each group.
- Beat the no-change baseline on mean detune response error.
- Keep mean absolute measurement error within 10% of the uniform arm on the common evaluation measurements.
- Finish all requested updates within resource limits, with finite losses/gradients and verified data identities.

Report both arms and every group even when a gate fails. Relative improvement over the uniform arm is descriptive; the absolute no-change comparison prevents a large aggregate gain from hiding an unlearned envelope group. A failed check does not prove that no model could learn these sounds or that longer training could never help. It means this bounded learning mechanism has not earned another full run without a specific evidence-based revision.

Limits: **2,000 update attempts total, zero native requests, 30 minutes, 8 GiB process-tree RAM and 256 MiB new output files**, float32 CPU with two threads. Every attempted update is durably charged before execution; lost work is not a free retry. Resume requires the unchanged plan, identities and checkpoints and retains the original wall-time/update caps. The repaired arm processes more frames, so equal updates do not imply equal computation; report each arm's elapsed time.

Do not export a qualified helper or change the inverse predictor. Both historical failed helpers remain refused. Passing this fitting check would still require the full native ranking/guidance gates and later independent confirmation/listening. Keep the 14-output patch vocabulary and local-only execution.

Prepare with `run_event_response_fit.py --parent COMPLETED_PAIRED_RUN --repaired COMPLETED_EVENT_PREFLIGHT --output NEW_DIRECTORY`; execute the unchanged plan with `--execute`. Use `--execute --resume` only after verifying the original process has stopped and the unchanged budgets permit recovery.

Frozen plan ID: `93d1046bfb6ed9571f9b41c771720657107d6777aee9e07524a7bfc73d2e0dad`. All 48 input files (24 per representation) passed hash/recipe checks before fitting. Twenty focused tests pass, including small-pool exact recovery and a gate check that refuses a failed release group despite otherwise favorable averages. Existing full-screen pair-count rejection remains tested.
