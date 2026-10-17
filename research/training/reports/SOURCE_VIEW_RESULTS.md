# Source and performance compatibility

The new opt-in [attack-compatible sampler](../SOURCE_PERFORMANCE_SAMPLING.md) keeps every training state and selects reference notes long enough for its stored amplitude delay and attack. It preserves source presets, targets, gain, spectral modes and amounts. Audio QA still decides whether the rendered result can enter training.

The [native plan](data/source_view_plan.json) and [summary](data/source_view_summary.json) cover the original frozen combined pool. All eight gates pass: expected native QA, exact fixed-phase repeats, original labels, identical presets between short/long views, the attack rule, all 205 training states in the schedule, unchanged source fingerprints and bounded memory. This is a source-view and recovery check, not a finished-model quality result.

| View | Native repeats | RMS | Interpretation |
|---|---|---|---|
| Original sequence 6 | Two expected rejections | 0 | Strong high-pass combination remains silent |
| Original sequence 15 | Two expected rejections | 0 | Strong high-pass combination remains silent |
| Original sequence 45 | Two expected rejections | `5.960843e-7` | 0.125-second notes do not expose its 0.143-second amplitude delay plus attack |
| Same sequence-45 patch, sampler-selected 1.5-second notes | Two accepted renders | `1.334629e-5` | A longer observation restores audible signal without changing the patch |

All eight attempts remain in the denominator: two accepted, six expected rejections, no errors. Correctness rendering fixes the three oscillator random phases and sampler random phase and uses a fresh process per request. Musical phase controls, seed bits, random modulation and all other source controls remain requested. Audio hashes match exactly within each repeated pair, including the rejected views.

The panel took 30.88 seconds after initial pool validation and schedule construction, with 1,005.81 MiB peak process-tree memory against an 8 GiB limit. Pool loading is excluded from that duration; it is not an end-to-end throughput benchmark. The environment is the same local Windows CPU setup as the [combined pipeline check](COMBINED_CURRICULUM_RESULTS.md). No waveforms, weights or factory asset payloads are committed.

The [39 exploratory interventions](data/source_view_diagnosis.json) preserve the causal investigation separately. Disabling spectral processing or reducing its amount restores the first two cases; merely lengthening their notes does not. The sampler-only case is restored by longer held notes or a faster envelope, while changes to disabled oscillator spectral controls do not help. Those exploratory results motivated the rule; the eight-attempt pinned panel independently confirms the unchanged-preset correction and expected negative fixtures. The high-pass range is not narrowed, and none of these state IDs is excluded from training.

The schedule has a new identity including its rule fingerprint. Recovery checks compare the uninterrupted and resumed model and verify saved performance IDs. Legacy sampler behavior remains unchanged. All 60 distinct selected local checks pass across the initial and corrected runs, covering the new sampler, existing split/recipe contracts, trainer recovery and paired training. The first run passed 59 checks and exposed one incorrect test expectation about the existing float32 envelope conversion. The corrected five-test sampler run passed in 107.04 seconds; it compares the complete target exactly before/after eligibility and accounts for float32 conversion when checking authored decimal values. Ruff, whitespace and all 61 pinned source fingerprints pass.

Full source robustness still requires broader views and actual QA accounting. Enough attack time alone does not make filtering, modulation or every parameter audible. Checkpoint persistence is the next measured bottleneck, followed by input-layout and active graph/range gaps, data scaling and useful validation quality. Final-test audio stays reserved, Actions stays disabled and no model is promoted.
