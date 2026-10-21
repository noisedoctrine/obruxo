# Training-plan completion ledger

Updated 2026-10-22 for the [spectral reconstruction plan](../modelling/TRAINING_PLAN.md). Sound-loss training moves into the first small stage; parameter matching is an initialization/auxiliary signal. Codec coverage and model quality remain separate. Upcoming work is local only; Colab work is deferred. On-demand audio, frozen test reservations and local checks with GitHub Actions disabled remain in force. PRs are reviewed, never automatically merged.

## Active training track

| Stage | Evidence | Remaining acceptance work |
|---|---|---|
| A0: spectral/control contract | [Local contract check](reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md): unpooled targets/loss, Gumbel adapter, hard export, gradient/recovery checks; 33 tests, 12 native requests, six exact repeats and memory profile; [A1 card frozen](reports/BASIC_SPECTRAL_HELPER_CARD.md) | Complete for this bounded implementation |
| A1: useful sound-loss gradients | [Settled-renderer qualification](reports/SETTLED_NATIVE_GRADIENT_RESULTS.md) passes: 22/24 continuous and 21/24 joint improvements, 560 accepted requests and 24 exact repeats. Both learned helpers remain unqualified | Complete for this bounded native estimator/profile |
| A2: spectral-loss inverse learning | [64-state trial](reports/BROADER_NATIVE_INVERSE_RESULTS.md) complete and exactly replayed; full/original fit passes, held-out reference +5.82% / transfer −0.24% miss quality target | Check paired cold/settled input records and historical initialization before another fit; no promotion |
| A3: basic quality and bundle | [21-case](reports/STATE_COVERAGE_RESULTS.md) and [23-case](reports/FRESH_VALIDATION_RESULTS.md) development evidence; existing narrow inference exports | Confirm fresh families/performances, second seed, listening, recovery and reproducible basic bundle |
| B: bounded sound expansion | Existing broader components can be reused | Choose one missing capability after A passes; allocate only its required heads; demonstrate gain without earlier-stage regression |
| C: simple modulation | Renderer/codec can support this | Train one LFO with at most two linear routes and a small destination list; evaluate sound, not exact graph identity |
| D: selective expansion and release | Offline narrow bundle/CLI and broad infrastructure exist | Admit useful capabilities individually, freeze supported scope, final test, fresh-environment bundle, listening/model card/runtime results |

Current task: **check paired cold/settled input features and the historical initial model before another fit**. The [completed broader trial](reports/BROADER_NATIVE_INVERSE_RESULTS.md) learns its fitted population but does not solve transfer. All 3,332 native requests and exact replay pass. Thirty-two matching patch/MIDI records with identical frontend/initial weights are available for a zero-render comparison. Do not assume initialization is the cause, expand coverage again automatically, extend the finished run, lower thresholds or add outputs. Both learned helpers remain unqualified; no model has been promoted.

The [new inverse card](reports/SETTLED_NATIVE_INVERSE_CARD.md) freezes plan `7fc5c19575d3869d3b82d9790b2c89d50c0e92f69ce702971d1446a69c8e5801`. Its [completed result](reports/SETTLED_NATIVE_INVERSE_RESULTS.md) stays within the original limits: 3,017 native attempts, 3 hours 25 minutes, 1.51 GiB RAM and 0.71 GiB archive. A battery-triggered host sleep caused one timeout after 106 updates; its checkpoint and ledgers are preserved, the identical request succeeds on bounded recovery, and all 128 updates replay exactly. The timeout remains counted. The development quality gates fail independently of that infrastructure interruption.

The [native inverse card](reports/NATIVE_INVERSE_FIT_CARD.md) now freezes 32 training states/two performances, 128 updates per arm and a conditional 12-state comparison. Its 3,328-request allocation explicitly accounts for numerical-gradient queries within the existing four-hour/8-GiB/2-GiB limits. Six focused training/inverse tests pass; the plan is frozen before execution. Run completion and reconstruction gains remain unproven.

That [run stopped on update 35](reports/NATIVE_INVERSE_FIT_RESULTS.md) when a trial four-voice configuration exceeded the audio amplitude limit. Its 34 committed updates, both optimizers and RNG replay exactly. No fit or validation quality result is available. The subsequent startup diagnosis below replaces the initial clipping-penalty hypothesis; the failed run and its original records remain unchanged.

The [startup diagnostic](reports/data/native_settling_summary.json) identifies a rendering defect: immediate MIDI starts while native controls settle. Delaying MIDI with an exactly aligned silent lead lowers the same peak from 1.032 to 0.244 and restores the quadratic level response. The tested penalty is not enabled. An 8,192-frame lead removed clipping but failed its unchanged settling tolerance; the longer profile below passes. Do not continue the old partial model or reuse its targets as corrected evidence. Exception summaries now retain committed progress; a regression test covers the original zero-count bug.

The [completed renderer check](reports/NATIVE_PREROLL_RESULTS.md) rejects the short 8,192-frame lead and qualifies **32,768 frames**, with zero difference versus a longer lead and exact repeats. This is now the sound-training default, with profile-aware qualification and target-cache rejection. The [settled native-gradient qualification](reports/SETTLED_NATIVE_GRADIENT_RESULTS.md) then passes in 17.83 minutes and 0.87 GiB RAM. Fresh inverse training uses new targets and optimizers under its own card; no model has been promoted.

The old helper's guided changes worsened real audio on all 20 cases despite lower predicted loss. It remains unqualified and disconnected. The new plan requires real-render gradient qualification before inverse integration. Existing supervised `train` commands do not become spectral-loss training merely because the plan changes.

The explicit `--prediction-scope basic` option rejects broad pools before creating the training frontend or renderer and records the choice in the run contract. It preserves the actual compact 14-value output layer. Older experiments retain their contracts through the default `codec` option.

Both existing basic validation panels have already influenced development. No test state is released for tuning. Full-model tiny-set fitting, eight-LFO loss tuning, all-64-slot training and mixed-corpus scaling are deferred; they do not block Stage A. Exact random-seed prediction and exhaustive parameter coverage are removed from completion criteria.

## Retained infrastructure and experiments

The following remain useful engineering evidence, not evidence that Stage A or release quality is complete:

- [Full native preset support](FULL_RENDER_PROFILE.md), [source ingestion](CORPUS_INPUTS.md), [long observations](LONG_OBSERVATIONS.md), [mixed-source streaming](MIXED_SOURCES.md), [checkpoint persistence](CHECKPOINT_STORAGE.md), frozen splits and QA logs.
- [Factory assets](FACTORY_ASSETS.md), [curve assets](LFO_BANK.md), [eight-curve training](LFO_TRAINING.md), [ordered routing](ORDERED_ROUTING.md), [custom curves and seeds](CUSTOM_CURVES_AND_SEEDS.md), [combined curriculum](COMBINED_CURRICULUM.md) and [full connection coverage](GRAPH_COVERAGE.md). Their large prediction heads are experimental, not defaults for the staged model.
- [Full-model fit diagnostic](reports/MIXED_OVERFIT_RESULTS.md): phase-wrap correction and valid native prediction checks; incomplete exact fit does not establish or refute narrow reconstruction quality.
- Paired identity, conditioning, sound-proxy, direct-render target and candidate-ranking experiments. Their bounded results are retained; admission to the active model requires evidence in the current stage.

The [superseded plan](../modelling/TRAINING_PLAN_COVERAGE_20261018.md) records the previous coverage milestones. Do not use its requirement to learn every control family as an active acceptance criterion.
