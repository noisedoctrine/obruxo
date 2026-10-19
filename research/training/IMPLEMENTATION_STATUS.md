# Training-plan completion ledger

Updated 2026-10-19 for the [staged reconstruction plan](../modelling/TRAINING_PLAN.md). Codec coverage and model quality are separate tracks. Free local/free Colab compute, on-demand audio, frozen test reservations and local checks with GitHub Actions disabled remain in force. PRs are reviewed, never automatically merged.

## Active training track

| Stage | Evidence | Remaining acceptance work |
|---|---|---|
| A: basic reconstruction | Nine continuous controls, two categorical choices; trained 205/404-state models; [21-case development results](reports/STATE_COVERAGE_RESULTS.md) and [23 additional cases](reports/FRESH_VALIDATION_RESULTS.md); recovery, native serialization and audio-only exports | Diagnose sound failures, make a justified correction, confirm held-out reference/transfer quality and listening usefulness |
| B: bounded sound expansion | Existing broader components can be reused | Choose one missing capability after A passes; allocate only its required heads; demonstrate gain without earlier-stage regression |
| C: simple modulation | Renderer/codec can support this | Train one LFO with at most two linear routes and a small destination list; evaluate sound, not exact graph identity |
| D: selective expansion and release | Offline narrow bundle/CLI and broad infrastructure exist | Admit useful capabilities individually, freeze supported scope, final test, fresh-environment bundle, listening/model card/runtime results |

Current task: test better stereo/width and detuning evidence in the unchanged basic output space. The [five-case native diagnosis](reports/BASIC_SOUND_DIAGNOSIS.md) completed all 70 requests and 10 exact phase-fixed repeat pairs. Correcting unison/detune/pan improves every case and lowers mean sound error by 26.68%; this uses known source controls and is not a trained-model gain. Freeze the next comparison under the [basic-stage protocol](STAGED_RECONSTRUCTION.md) before training; listening and confirmation remain open.

The explicit `--prediction-scope basic` option rejects broad pools before creating the training frontend or renderer and records the choice in the run contract. It preserves the actual compact 14-value output layer. Older experiments retain their contracts through the default `codec` option.

Both existing basic validation panels have already influenced development. No test state is released for tuning. Full-model tiny-set fitting, eight-LFO loss tuning, all-64-slot training and mixed-corpus scaling are deferred; they do not block Stage A. Exact random-seed prediction and exhaustive parameter coverage are removed from completion criteria.

## Retained infrastructure and experiments

The following remain useful engineering evidence, not evidence that Stage A or release quality is complete:

- [Full native preset support](FULL_RENDER_PROFILE.md), [source ingestion](CORPUS_INPUTS.md), [long observations](LONG_OBSERVATIONS.md), [mixed-source streaming](MIXED_SOURCES.md), [checkpoint persistence](CHECKPOINT_STORAGE.md), frozen splits and QA logs.
- [Factory assets](FACTORY_ASSETS.md), [curve assets](LFO_BANK.md), [eight-curve training](LFO_TRAINING.md), [ordered routing](ORDERED_ROUTING.md), [custom curves and seeds](CUSTOM_CURVES_AND_SEEDS.md), [combined curriculum](COMBINED_CURRICULUM.md) and [full connection coverage](GRAPH_COVERAGE.md). Their large prediction heads are experimental, not defaults for the staged model.
- [Full-model fit diagnostic](reports/MIXED_OVERFIT_RESULTS.md): phase-wrap correction and valid native prediction checks; incomplete exact fit does not establish or refute narrow reconstruction quality.
- Paired identity, conditioning, sound-proxy, direct-render target and candidate-ranking experiments. Their bounded results are retained; admission to the active model requires evidence in the current stage.

The [superseded plan](../modelling/TRAINING_PLAN_COVERAGE_20261018.md) records the previous coverage milestones. Do not use its requirement to learn every control family as an active acceptance criterion.
