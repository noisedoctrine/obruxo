# Training-plan completion ledger

Updated 2026-10-21 for the [spectral reconstruction plan](../modelling/TRAINING_PLAN.md). Sound-loss training moves into the first small stage; parameter matching is an initialization/auxiliary signal. Codec coverage and model quality remain separate. Upcoming work is local only; Colab work is deferred. On-demand audio, frozen test reservations and local checks with GitHub Actions disabled remain in force. PRs are reviewed, never automatically merged.

## Active training track

| Stage | Evidence | Remaining acceptance work |
|---|---|---|
| A0: spectral/control contract | [Local contract check](reports/BASIC_SPECTRAL_CONTRACT_RESULTS.md): unpooled targets/loss, Gumbel adapter, hard export, gradient/recovery checks; 33 tests, 12 native requests, six exact repeats and memory profile; [A1 card frozen](reports/BASIC_SPECTRAL_HELPER_CARD.md) | Complete for this bounded implementation |
| A1: useful sound-loss gradients | [Direct native-gradient qualification](reports/NATIVE_GRADIENT_RESULTS.md) passes: 24/24 continuous and 23/24 joint updates improve real audio, 560 accepted renders, 24 exact repeats. Both learned helpers remain unqualified | Complete for this bounded native estimator; query cost must be included in A2 |
| A2: spectral-loss inverse learning | Native objective connected to the actual inverse model; correctness/recovery tested. No spectral candidate trained yet | Freeze a costed card, train matched spectral/parameter-only arms, prove hard-export native gains and log total compute |
| A3: basic quality and bundle | [21-case](reports/STATE_COVERAGE_RESULTS.md) and [23-case](reports/FRESH_VALIDATION_RESULTS.md) development evidence; existing narrow inference exports | Confirm fresh families/performances, second seed, listening, recovery and reproducible basic bundle |
| B: bounded sound expansion | Existing broader components can be reused | Choose one missing capability after A passes; allocate only its required heads; demonstrate gain without earlier-stage regression |
| C: simple modulation | Renderer/codec can support this | Train one LFO with at most two linear routes and a small destination list; evaluate sound, not exact graph identity |
| D: selective expansion and release | Offline narrow bundle/CLI and broad infrastructure exist | Admit useful capabilities individually, freeze supported scope, final test, fresh-environment bundle, listening/model card/runtime results |

Current task: **cost and run the matched inverse-training stage using the qualified native-gradient path**. The [short helper fit](reports/EVENT_RESPONSE_FIT_RESULTS.md) failed; the [direct native alternative](reports/NATIVE_GRADIENT_RESULTS.md) subsequently passed its separately frozen gates in 20 minutes and 0.88 GiB RAM. Budget the small fit and comparison explicitly before execution: each example can need 22 gradient queries. Preserve the 14-output scope and existing checkpoint as the shared starting point. Both learned helpers remain unqualified; no new inverse model has been promoted.

The [native inverse card](reports/NATIVE_INVERSE_FIT_CARD.md) now freezes 32 training states/two performances, 128 updates per arm and a conditional 12-state comparison. Its 3,328-request allocation explicitly accounts for numerical-gradient queries within the existing four-hour/8-GiB/2-GiB limits. Six focused training/inverse tests pass; the plan is frozen before execution. Run completion and reconstruction gains remain unproven.

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
