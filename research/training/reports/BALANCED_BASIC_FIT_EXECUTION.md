# Balanced three-sound fit: completed execution

The [frozen fit card](BALANCED_BASIC_FIT_CARD.md) is complete. The run and independent audit both completed, with no reserved/test data used. **The candidate failed the existing admission gates and was not retained as a model.** The separate [label-only learnability check](BALANCED_SUPERVISED_CONTROL_RESULTS.md) also completed; its source-label loss improved while real-Vital sound quality worsened.

## Run record

| Item | Result |
|---|---|
| Frozen plan | `63e8b02e659bb5b8c78c8f1e2810b1eae23df9a1778fc652316efcc9c0e3d69c` |
| Optimizer updates | 48 committed; 46 accepted by native sound checks |
| Native attempts | 6,292 of 8,000 maximum |
| Wall time | 12,252 seconds (3 h 24 min) |
| Peak process-tree memory | 1.57 GiB; under 4 GiB |
| Independent replay and audit | Passed; model, Adam state, trace and RNG reproduced; 426 comparison rows rescored |
| Reserved/test data | Not used |
| Model admission | **Failed** |

## Why it failed

The fit did improve the original-input own/transfer groups, but failed the full multi-group requirements:

- All three source sounds failed the per-sound learning prerequisite.
- Only one of the three added pitch-grid groups passed.
- The wider alternate-input own/transfer groups failed.
- Earlier-sound preservation failed.
- The full reconstruction and directional-benefit gates both failed.

These failures are retained as measured; the evaluation gates were not relaxed. The balanced fit remains a completed development run, not a model candidate.

The [data bank report](BALANCED_BASIC_DATA_RESULTS.md) remains valid. The label-only and four-schedule comparisons are now complete as separate diagnostics; neither changes this failed native sound-fit result. See the [current training plan](../../modelling/TRAINING_PLAN.md) for the next step: diagnose this audited fit's group-level failures before another long native run. The independent 192-root reserve and final test remain unscored; Actions stays disabled and Colab deferred.
