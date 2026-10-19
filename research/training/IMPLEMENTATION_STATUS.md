# Training-plan completion ledger

Updated 2026-10-19 for the [spectral reconstruction plan](../modelling/TRAINING_PLAN.md). Sound-loss training moves into the first small stage; parameter matching is an initialization/auxiliary signal. Codec coverage and model quality remain separate. Free local/free Colab compute, on-demand audio, frozen test reservations and local checks with GitHub Actions disabled remain in force. PRs are reviewed, never automatically merged.

## Active training track

| Stage | Evidence | Remaining acceptance work |
|---|---|---|
| A0: spectral/control contract | Basic 14-value head, supervised 205/404-state models, recovery and codec; [native diagnosis](reports/BASIC_SOUND_DIAGNOSIS.md) complete | Implement/version spectral targets, differentiable adapter and Gumbel-softmax; verify sound-only gradients, hard export, frozen helper and recovery; profile compute |
| A1: useful helper gradients | Collection/recovery infrastructure exists; [old alpha helper failed](reports/SOUND_PROXY_RESULTS.md) | Qualify a new basic-scope helper on held-out performances, native rankings and actual continuous/category-guided improvements |
| A2: spectral-loss inverse learning | Not implemented/qualified by this revision | Freeze qualified helper, train matched spectral/parameter-only arms, prove hard-export native gains and log total compute |
| A3: basic quality and bundle | [21-case](reports/STATE_COVERAGE_RESULTS.md) and [23-case](reports/FRESH_VALIDATION_RESULTS.md) development evidence; existing narrow inference exports | Confirm fresh families/performances, second seed, listening, recovery and reproducible basic bundle |
| B: bounded sound expansion | Existing broader components can be reused | Choose one missing capability after A passes; allocate only its required heads; demonstrate gain without earlier-stage regression |
| C: simple modulation | Renderer/codec can support this | Train one LFO with at most two linear routes and a small destination list; evaluate sound, not exact graph identity |
| D: selective expansion and release | Offline narrow bundle/CLI and broad infrastructure exist | Admit useful capabilities individually, freeze supported scope, final test, fresh-environment bundle, listening/model card/runtime results |

Current task: **A0, the small spectral-training contract and gradient path**, under the [basic-stage protocol](STAGED_RECONSTRUCTION.md). This plan revision launches no experiments. The five-case diagnosis completed 70 requests and 10 exact phase-fixed repeat pairs; correcting unison/detune/pan lowers mean error 26.68%, using known source controls. Those findings guide spectral/stereo sensitivity checks, not a new parameter-only feature sweep.

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
