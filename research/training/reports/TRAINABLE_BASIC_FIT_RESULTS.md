# Learning audio features preserves earlier gains but still misses held-note quality

The [completed fit](data/trainable_basic_fit_summary.json) passes four of six reconstruction groups. Both held-note input groups remain worse than the matched parameter-trained control, so **A2 remains incomplete and neither model is promoted**. All recording types in this small panel were used for training; this is not independent validation.

Both arms learn the copied audio-feature branches while keeping the original predictor and Basic Pitch frontend frozen. The model retains 14 dense outputs / 11 codec selections, 613,788 total weights and 257,982 trainable weights. The sound candidate receives measured sound loss and reference rehearsal, with no parameter-label auxiliary. The [original card](TRAINABLE_BASIC_FIT_CARD.md), initializer, 48-update order and every gate remain unchanged.

| Input recording / playback | Combined loss versus parameter control | MRSTFT versus parameter control | Frozen group gate |
|---|---:|---:|---|
| Original / same performance | -41.97% | -40.33% | Pass |
| Original / changed performance | -40.16% | -40.95% | Pass |
| Alternate / same performance | -13.80% | -7.44% | Pass |
| Alternate / changed performance | -3.80% | +2.85% | Pass |
| Held note / same performance | +8.06% | +6.83% | **Fail** |
| Held note / changed performance | +11.08% | +9.18% | **Fail** |

Negative values mean lower error. The gate uses the frozen combined objective and its other criteria; a passing group does not imply that every reported metric beats the control. All six combined-loss means improve against initialization. The unchanged earlier-32 preservation gate now passes: combined loss improves **1.85%**, MRSTFT **3.29%** and the MRSTFT 90th percentile **1.62%** against the stronger earlier sound model. These are small-set results, not a claim of perceptual usefulness or a proven general feature-learning benefit.

The [independent audit](data/trainable_basic_fit_verification.json) exactly replays both 48-update model/Adam/RNG trajectories, all 5,803 saved training scalar probes and the frozen parameter-control rehearsal. Both copied input branches change in each arm. All **312 retained comparison waveforms** are independently rescored and both exports cold-load correctly. The sound arm accepts 44/48 proposed updates; refused weights roll back under the original rule. Training probe audio is not independently rescored.

All **6,011 native requests** complete without recovery. Native work takes 3 h 23 min, with **1.12 GiB peak process-tree RAM** and 0.564 GiB retained evidence before the summary, within the six-hour / 4-GiB / 2-GiB limits. The separate audit takes 285.70 seconds / 0.90 GiB and adds no renders or experimental updates. Both [training](data/trainable_basic_fit_service_001_completion.json) and [audit](data/trainable_basic_fit_service_002_completion.json) drivers exit successfully, release their scoped awake requests and preserve checkpoints. The [queued follow-up](data/trainable_basic_fit_audit_followup.json) confirms verification. No test state is evaluated.

Keep this candidate diagnostic. Inspect the saved held-note failures and prepare representative listening examples before selecting another structural change. Do not add outputs, extend this fit, sweep thresholds or run another parameter-only initializer. The [192-family reservation](BASIC_CONFIRMATION_RESERVATION_RESULTS.md) stays unscored; the failed A2 result authorizes no confirmation evaluation or second seed. Full A2/A3, listening and release requirements remain open.
